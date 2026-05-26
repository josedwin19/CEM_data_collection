#!/usr/bin/env python3
"""
EPU Session Metadata Extractor
================================
Parses an EPU cryo-EM data collection session directory and generates
a structured acquisition report.

Usage:
    python Data_Collection.py /path/to/session_dir [options]

Options:
    --output, -o  Output file path (.txt or .json). Default: <session_dir>/epu_report.txt
    --format, -f  Output format: 'text' or 'json'. Default: 'text'
    --max-movies  Max movies to parse for dose averaging. Default: 500

Expected directory layout (standard EPU output):
    session_dir/
    ├── EpuSession.dm              ← session-level metadata
    ├── Preset_*.sxml              ← acquisition presets
    ├── Metadata/
    │   └── GridSquare_*.dm
    └── Images-Disc1/
        └── GridSquare_XXXXXXX/
            ├── Data/
            │   ├── FoilHole_*_Fractions.xml  ← per-movie dose & camera data
            │   └── FoilHole_*.xml            ← per-movie optics (voltage, Cs, mag)
            └── FoilHoles/
"""

import os
import re
import sys
import glob
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, stdev
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_first(pattern: str, text: str, flags=0) -> Optional[str]:
    """Return first captured group of a regex, or None."""
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def safe_float(val: Optional[str]) -> Optional[float]:
    try:
        return float(val) if val else None
    except (ValueError, TypeError):
        return None


def ns_strip(tag: str) -> str:
    """Strip XML namespace from a tag string."""
    return re.sub(r'\{[^}]+\}', '', tag)


def xml_find_text(root, *paths) -> Optional[str]:
    """Try multiple tag paths (ignoring namespaces) and return first found text."""
    flat = ET.tostring(root, encoding='unicode')
    for path in paths:
        # Use regex to find tags regardless of namespace prefixes
        m = re.search(rf'<[^>]*:?{re.escape(path)}[^>]*>([^<]+)<', flat, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_epu_session(session_path: Path) -> dict:
    """Extract session-level info from EpuSession.dm."""
    dm_file = session_path / 'EpuSession.dm'
    result = {}

    if not dm_file.exists():
        print(f"[WARNING] EpuSession.dm not found at {dm_file}", file=sys.stderr)
        return result

    content = dm_file.read_text(encoding='utf-8', errors='replace')

    # Session name
    name = find_first(r'<Name[^>]*>([^<]+)</Name>', content)
    result['session_name'] = name

    # EPU software version
    version = find_first(r'z:Assembly="Applications\.Epu\.Persistence, Version=([\d.]+)', content)
    result['epu_version'] = version

    # Phase plate
    pp = find_first(r'<PhasePlateEnabled>([^<]+)</PhasePlateEnabled>', content)
    result['phase_plate_enabled'] = pp

    # AutoLoader slot
    slot = find_first(r'<AutoloaderSlot>([^<]+)</AutoloaderSlot>', content)
    result['autoloader_slot'] = slot

    return result


def parse_preset_sxml(session_path: Path) -> dict:
    """Extract acquisition preset settings from Preset_*.sxml."""
    result = {}
    sxml_files = list(session_path.glob('Preset_*.sxml'))
    if not sxml_files:
        return result

    sxml_file = sxml_files[0]
    result['preset_file'] = sxml_file.name
    content = sxml_file.read_text(encoding='utf-8', errors='replace')

    # Locate the "Data Acquisition" preset block
    idx = content.find('Data Acquisition')
    if idx < 0:
        return result
    block = content[idx: idx + 12000]

    # Number of fractions (count DoseFraction definitions)
    n_fracs = len(re.findall(r'<c:EndFrameNumber>', block))
    result['fractions_from_preset'] = n_fracs if n_fracs > 0 else None

    # Exposure time (total, in seconds)
    et_vals = re.findall(r'<b:ExposureTime>([^<]+)</b:ExposureTime>', block)
    # The first value is typically the total exposure; filter out '1' (autofocus/etc.)
    et_filtered = [float(v) for v in et_vals if float(v) > 2]
    if et_filtered:
        result['total_exposure_time_s'] = et_filtered[0]

    # Electron counting
    ec = find_first(
        r'ElectronCountingEnabled.*?<Value[^>]*>([^<]+)</Value>', block, re.DOTALL
    )
    result['electron_counting'] = ec

    # Super-resolution factor
    sr = find_first(
        r'SuperResolutionFactor.*?<Value[^>]*>([^<]+)</Value>', block, re.DOTALL
    )
    result['super_resolution_factor'] = sr

    return result


def parse_defocus_txt(session_path: Path) -> dict:
    """Read Defocus_*.txt (user-provided notes)."""
    result = {}
    txt_files = list(session_path.glob('Defocus*.txt')) + list(session_path.glob('defocus*.txt'))
    if not txt_files:
        return result

    text = txt_files[0].read_text(encoding='utf-8', errors='replace').strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Parse comma-separated defocus values (µm)
    for line in lines:
        values = re.findall(r'-?\d+\.?\d*', line)
        if len(values) >= 2:
            floats = [float(v) for v in values]
            result['defocus_range_um'] = f"{min(floats):.1f} to {max(floats):.1f}"
            result['defocus_values_um'] = [float(v) for v in values]
            break

    # Objective aperture
    ap = find_first(r'(\d+)\s*um', text, re.IGNORECASE)
    if ap:
        result['objective_aperture_um'] = int(ap)

    return result


def parse_fractions_xml(xml_path: Path) -> Optional[dict]:
    """
    Parse a single FoilHole_*_Fractions.xml file.
    Returns per-movie metadata dict, or None on failure.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[WARNING] Could not parse {xml_path.name}: {e}", file=sys.stderr)
        return None

    content = ET.tostring(root, encoding='unicode')
    result = {'source_file': xml_path.name}

    # Camera info
    result['camera_commercial_name'] = xml_find_text(root, 'CommercialName')
    result['camera_internal_name'] = xml_find_text(root, 'CameraName')

    # Electron counting
    ec = xml_find_text(root, 'ElectronCounting')
    result['electron_counting'] = ec

    # Image dimensions
    result['image_width_px'] = safe_float(xml_find_text(root, 'Width'))
    result['image_height_px'] = safe_float(xml_find_text(root, 'Height'))

    # Sensor pixel size → calibrated pixel size at sample (in Ångström)
    sensor_w = find_first(r'<SensorPixelSize>.*?<Width>([^<]+)</Width>', content, re.DOTALL)
    if sensor_w:
        px_m = float(sensor_w)
        result['pixel_size_A'] = round(px_m * 1e10, 4)
        result['pixel_size_m'] = px_m

    # Fractions
    result['expected_fractions'] = safe_float(xml_find_text(root, 'ExpectedNumberOfFractions'))
    result['recorded_fractions'] = safe_float(xml_find_text(root, 'RecordedNumberOfFractions'))

    # Acquisition time
    acq_time = xml_find_text(root, 'Time')
    result['acquisition_time'] = acq_time

    # Per-fraction data → total accumulated dose
    doses = []
    exp_times = []
    for frac in root.findall('.//{*}Fraction') or root.findall('.//Fraction'):
        dose_el = frac.find('{*}TotalDose') or frac.find('TotalDose')
        et_el   = frac.find('{*}ExposureTime') or frac.find('ExposureTime')
        if dose_el is not None and dose_el.text:
            doses.append(float(dose_el.text))
        if et_el is not None and et_el.text:
            exp_times.append(float(et_el.text))

    if doses:
        result['total_dose_e_A2'] = round(sum(doses), 3)
        result['dose_per_fraction_e_A2'] = round(mean(doses), 4)
    if exp_times:
        result['total_exposure_time_s'] = round(sum(exp_times), 3)

    return result


def parse_movie_xml(xml_path: Path) -> Optional[dict]:
    """
    Parse the non-Fractions FoilHole_*.xml to extract microscope optics
    (AccelerationVoltage, SphericalAberration, NominalMagnification, PixelSize).
    """
    try:
        content = xml_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return None

    result = {}

    # Acceleration voltage (stored in Volts)
    voltage = find_first(r'AccelerationVoltage[^>]*>([^<]+)<', content, re.IGNORECASE)
    if voltage:
        v = safe_float(voltage)
        if v and v > 1000:
            result['acceleration_voltage_kV'] = round(v / 1000, 0)

    # Spherical aberration Cs (metres → mm)
    cs = find_first(r'<[^>]*SphericalAberration[^>]*>([^<]+)<', content, re.IGNORECASE)
    if cs:
        cs_val = safe_float(cs)
        if cs_val is not None:
            result['Cs_mm'] = round(cs_val * 1000, 2) if cs_val < 0.1 else cs_val

    # Nominal magnification
    mag = find_first(r'<[^>]*NominalMagnification[^>]*>(\d+)<', content, re.IGNORECASE)
    if mag and int(mag) > 1000:
        result['nominal_magnification'] = int(mag)

    # Calibrated pixel size
    px = find_first(r'<[^>]*PixelSize[^>]*>([^<]+)<', content, re.IGNORECASE)
    if px:
        px_val = safe_float(px)
        if px_val and px_val < 1e-8:
            result['pixel_size_A_optics'] = round(px_val * 1e10, 4)

    return result if result else None


# ──────────────────────────────────────────────────────────────────────────────
# Main collector
# ──────────────────────────────────────────────────────────────────────────────

def collect_metadata(session_path: Path, max_movies: int = 500) -> dict:
    """
    Walk the EPU session directory and aggregate all metadata.
    """
    metadata = {
        'session_path': str(session_path),
        'extraction_timestamp': datetime.now().isoformat(),
    }

    print(f"[INFO] Parsing session: {session_path}", file=sys.stderr)

    # 1 ── Session-level info
    metadata['session'] = parse_epu_session(session_path)

    # 2 ── Preset settings
    metadata['preset'] = parse_preset_sxml(session_path)

    # 3 ── Defocus / notes
    metadata['defocus'] = parse_defocus_txt(session_path)

    # 4 ── Scan for Images-Disc* folders
    disc_dirs = sorted(session_path.glob('Images-Disc*'))
    if not disc_dirs:
        print("[WARNING] No Images-Disc* directory found.", file=sys.stderr)

    all_fractions_xmls = []
    all_movie_xmls = []
    grid_square_count = 0
    foil_hole_count = 0

    for disc_dir in disc_dirs:
        for gs_dir in sorted(disc_dir.iterdir()):
            if not gs_dir.is_dir() or not gs_dir.name.startswith('GridSquare'):
                continue
            grid_square_count += 1

            data_dir = gs_dir / 'Data'
            if data_dir.exists():
                fracs = sorted(data_dir.glob('FoilHole_*_Fractions.xml'))
                movies = [
                    f for f in sorted(data_dir.glob('FoilHole_*.xml'))
                    if '_Fractions' not in f.name
                ]
                all_fractions_xmls.extend(fracs)
                all_movie_xmls.extend(movies)

            # Count foil holes (from FoilHoles subfolder)
            fh_dir = gs_dir / 'FoilHoles'
            if fh_dir.exists():
                fh_xmls = list(fh_dir.glob('FoilHole_*.xml'))
                foil_hole_count += len(fh_xmls)

    metadata['counts'] = {
        'grid_squares': grid_square_count,
        'foil_holes_with_xml': foil_hole_count,
        'movies_fractions_xml': len(all_fractions_xmls),
        'movies_optics_xml': len(all_movie_xmls),
    }

    print(f"[INFO] Found {len(all_fractions_xmls)} Fractions XMLs across "
          f"{grid_square_count} grid squares.", file=sys.stderr)

    # 5 ── Parse Fractions XMLs (cap at max_movies)
    per_movie = []
    sample_files = all_fractions_xmls[:max_movies]
    for xml_path in sample_files:
        data = parse_fractions_xml(xml_path)
        if data:
            per_movie.append(data)

    metadata['per_movie_sample_count'] = len(per_movie)

    # 6 ── Aggregate per-movie data
    if per_movie:
        # Camera info (take from first successful parse)
        first = per_movie[0]
        metadata['camera'] = {
            'commercial_name': first.get('camera_commercial_name'),
            'internal_name': first.get('camera_internal_name'),
            'electron_counting': first.get('electron_counting'),
            'image_width_px': first.get('image_width_px'),
            'image_height_px': first.get('image_height_px'),
        }
        metadata['pixel_size_A'] = first.get('pixel_size_A')
        metadata['fractions_per_movie'] = int(first.get('recorded_fractions', 0) or 0)

        # Dose statistics across all movies
        doses = [m['total_dose_e_A2'] for m in per_movie if m.get('total_dose_e_A2')]
        if doses:
            metadata['dose_stats'] = {
                'average_total_dose_e_A2': round(mean(doses), 2),
                'stdev_total_dose_e_A2': round(stdev(doses), 2) if len(doses) > 1 else 0,
                'min_total_dose_e_A2': round(min(doses), 2),
                'max_total_dose_e_A2': round(max(doses), 2),
                'n_movies_used': len(doses),
            }
            dose_per_frac = [m['dose_per_fraction_e_A2'] for m in per_movie
                             if m.get('dose_per_fraction_e_A2')]
            if dose_per_frac:
                metadata['dose_stats']['avg_dose_per_fraction_e_A2'] = round(mean(dose_per_frac), 4)

        # Acquisition timestamps
        times = [m['acquisition_time'] for m in per_movie if m.get('acquisition_time')]
        if times:
            metadata['first_movie_time'] = times[0]
            metadata['last_movie_time'] = times[-1]

    # 7 ── Parse optics from non-Fractions movie XMLs (first available)
    optics = {}
    for xml_path in all_movie_xmls[:20]:
        result = parse_movie_xml(xml_path)
        if result:
            optics.update({k: v for k, v in result.items() if v is not None})
            if 'acceleration_voltage_kV' in optics and 'Cs_mm' in optics:
                break

    metadata['optics'] = optics

    return metadata


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def format_text_report(meta: dict) -> str:
    lines = []
    div = "=" * 60

    lines.append(div)
    lines.append("  EPU SESSION — DATA COLLECTION REPORT")
    lines.append(div)
    lines.append(f"  Generated: {meta.get('extraction_timestamp', 'N/A')}")
    lines.append(f"  Session path: {meta.get('session_path', 'N/A')}")
    lines.append("")

    # Session info
    sess = meta.get('session', {})
    lines.append("── SESSION ──────────────────────────────────────────")
    lines.append(f"  Session name    : {sess.get('session_name', 'N/A')}")
    lines.append(f"  EPU version     : {sess.get('epu_version', 'N/A')}")
    lines.append(f"  Phase plate     : {sess.get('phase_plate_enabled', 'N/A')}")
    lines.append(f"  AutoLoader slot : {sess.get('autoloader_slot', 'N/A')}")
    lines.append("")

    # Counts
    counts = meta.get('counts', {})
    lines.append("── COLLECTION SUMMARY ───────────────────────────────")
    lines.append(f"  Grid squares collected : {counts.get('grid_squares', 'N/A')}")
    lines.append(f"  Foil holes (with XML)  : {counts.get('foil_holes_with_xml', 'N/A')}")
    lines.append(f"  Movies (total)         : {counts.get('movies_fractions_xml', 'N/A')}")
    if meta.get('first_movie_time') or meta.get('last_movie_time'):
        lines.append(f"  First acquisition  : {meta.get('first_movie_time', 'N/A')}")
        lines.append(f"  Last acquisition   : {meta.get('last_movie_time', 'N/A')}")
    lines.append("")

    # Microscope optics
    optics = meta.get('optics', {})
    lines.append("── MICROSCOPE OPTICS ────────────────────────────────")
    kV = optics.get('acceleration_voltage_kV')
    lines.append(f"  Acceleration voltage : {f'{int(kV)} kV' if kV else 'N/A (not in provided files)'}")
    cs = optics.get('Cs_mm')
    lines.append(f"  Spherical aberration : {f'{cs} mm' if cs else 'N/A (not in provided files)'}")
    mag = optics.get('nominal_magnification')
    lines.append(f"  Nominal magnification: {f'{mag:,}x' if mag else 'N/A (not in provided files)'}")
    lines.append("")

    # Camera & detector
    cam = meta.get('camera', {})
    lines.append("── DETECTOR ─────────────────────────────────────────")
    lines.append(f"  Detector          : {cam.get('commercial_name', 'N/A')}")
    lines.append(f"  Camera name       : {cam.get('internal_name', 'N/A')}")
    lines.append(f"  Electron counting : {cam.get('electron_counting', 'N/A')}")
    sr = meta.get('preset', {}).get('super_resolution_factor', '1')
    lines.append(f"  Super-resolution  : {'Yes' if sr and sr != '1' else 'No'} (factor: {sr or '1'})")
    w = cam.get('image_width_px')
    h = cam.get('image_height_px')
    if w and h:
        lines.append(f"  Image size        : {int(w)} × {int(h)} pixels")
    lines.append("")

    # Acquisition parameters
    lines.append("── ACQUISITION PARAMETERS ───────────────────────────")
    px = meta.get('pixel_size_A')
    lines.append(f"  Pixel size        : {f'{px} Å' if px else 'N/A'}")
    nf = meta.get('fractions_per_movie')
    pr_nf = meta.get('preset', {}).get('fractions_from_preset')
    lines.append(f"  Factions/movie   : {nf or pr_nf or 'N/A'}")
    et = meta.get('preset', {}).get('total_exposure_time_s')
    lines.append(f"  Total exposure    : {f'{et:.2f} s' if et else 'N/A'}")
    lines.append("")

    # Dose
    dose_stats = meta.get('dose_stats', {})
    lines.append("── DOSE ─────────────────────────────────────────────")
    avg = dose_stats.get('average_total_dose_e_A2')
    lines.append(f"  Avg total dose/movie    : {f'{avg:.2f} e⁻/Å²' if avg else 'N/A'}")
    dpf = dose_stats.get('avg_dose_per_fraction_e_A2')
    lines.append(f"  Avg dose/fraction       : {f'{dpf:.4f} e⁻/Å²' if dpf else 'N/A'}")
    sd = dose_stats.get('stdev_total_dose_e_A2')
    lines.append(f"  Std dev (total dose)    : {f'{sd:.2f} e⁻/Å²' if sd else 'N/A'}")
    mn = dose_stats.get('min_total_dose_e_A2')
    mx = dose_stats.get('max_total_dose_e_A2')
    if mn and mx:
        lines.append(f"  Dose range              : {mn:.2f} – {mx:.2f} e⁻/Å²")
    n = dose_stats.get('n_movies_used')
    lines.append(f"  Movies used for avg     : {n or meta.get('per_movie_sample_count', 'N/A')}")
    lines.append("")

    # Defocus
    df = meta.get('defocus', {})
    lines.append("── DEFOCUS & OPTICS SETTINGS ────────────────────────")
    lines.append(f"  Defocus range     : {df.get('defocus_range_um', 'N/A')} µm")
    
    obj_ap = df.get('objective_aperture_um')
    lines.append(f"  Objective aperture: {obj_ap} µm" if obj_ap else "  Objective aperture: N/A")
    lines.append("")

    # Preset file
    pr = meta.get('preset', {})
    if pr.get('preset_file'):
        lines.append("── PRESET FILE ──────────────────────────────────────")
        lines.append(f"  File: {pr.get('preset_file')}")
        lines.append("")

    lines.append(div)
    lines.append("  NOTE: Acceleration voltage, Cs, and nominal")
    lines.append("  magnification are read from the per-movie")
    lines.append("  FoilHole_*.xml (non-Fractions). If those files")
    lines.append("  are absent, values will show as N/A.")
    lines.append(div)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from an EPU cryo-EM session and generate a report."
    )
    parser.add_argument(
        'session_dir',
        help='Root path of the EPU session (contains EpuSession.dm, Images-Disc1/, etc.)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output file path. Defaults to <session_dir>/epu_report.txt'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format: text (human-readable) or json (machine-readable). Default: text'
    )
    parser.add_argument(
        '--max-movies',
        type=int,
        default=500,
        help='Maximum number of Fractions XMLs to parse for dose averaging. Default: 500'
    )
    args = parser.parse_args()

    session_path = Path(args.session_dir).resolve()
    if not session_path.exists():
        print(f"[ERROR] Session directory not found: {session_path}", file=sys.stderr)
        sys.exit(1)

    # Collect
    metadata = collect_metadata(session_path, max_movies=args.max_movies)

    # Format output
    if args.format == 'json':
        output_text = json.dumps(metadata, indent=2, ensure_ascii=False)
        default_ext = 'json'
    else:
        output_text = format_text_report(metadata)
        default_ext = 'txt'

    # Write or print
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = session_path / f'epu_report.{default_ext}'

    out_path.write_text(output_text, encoding='utf-8')
    print(f"[INFO] Report written to: {out_path}", file=sys.stderr)
    print(output_text)


if __name__ == '__main__':
    main()
