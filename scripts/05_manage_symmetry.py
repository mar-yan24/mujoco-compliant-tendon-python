import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
import argparse

def get_muscles(xml_path):
    print(f"Parsing: {xml_path}")
    if not os.path.exists(xml_path):
        print(f"Error: File not found: {xml_path}")
        return None, None, None

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return None, None, None

    root = tree.getroot()
    
    # Handle wrapping
    actuators = []
    # If the root has an 'actuator' child, iterate its children
    actuator_tag = root.find('actuator')
    if actuator_tag is not None:
        actuators = list(actuator_tag)
    else:
        # Otherwise assume the root contains the list (or root is the list)
        actuators = list(root)

    # Dictionary to store muscles
    muscles_r = {}
    muscles_l = {}
    
    for elem in actuators:
        name = elem.get('name')
        if not name:
            continue
        
        # Check if it has _r or _l suffix
        if name.endswith('_r'):
            key = name[:-2]
            muscles_r[key] = elem
        elif name.endswith('_l'):
            key = name[:-2]
            muscles_l[key] = elem

    return tree, muscles_r, muscles_l

def check_symmetry(xml_path):
    tree, muscles_r, muscles_l = get_muscles(xml_path)
    if tree is None: return

    print(f"Found {len(muscles_r)} right-side muscles and {len(muscles_l)} left-side muscles.")

    mismatches = []
    all_keys = set(muscles_r.keys()) | set(muscles_l.keys())
    target_attrs = ['biasprm', 'gainprm', 'lengthrange']
    
    for key in sorted(all_keys):
        if key not in muscles_r:
            mismatches.append(f"Missing right muscle for: {key}_l")
            continue
        if key not in muscles_l:
            mismatches.append(f"Missing left muscle for: {key}_r")
            continue
            
        r_elem = muscles_r[key]
        l_elem = muscles_l[key]
        
        for attr in target_attrs:
            val_r = r_elem.get(attr)
            val_l = l_elem.get(attr)
            
            if val_r is None and val_l is None:
                continue
            if val_r is None:
                mismatches.append(f"[{key}] Attribute '{attr}' missing in Right side")
                continue
            if val_l is None:
                mismatches.append(f"[{key}] Attribute '{attr}' missing in Left side")
                continue
                
            try:
                arr_r = np.array([float(x) for x in val_r.split()])
                arr_l = np.array([float(x) for x in val_l.split()])
                
                if arr_r.shape != arr_l.shape:
                    mismatches.append(f"[{key}] {attr} shape mismatch")
                elif not np.allclose(arr_r, arr_l, atol=1e-5):
                     diff = np.abs(arr_r - arr_l)
                     max_diff = np.max(diff)
                     mismatches.append(f"[{key}] {attr} mismatch (max diff: {max_diff:.6f})\n      R: {val_r}\n      L: {val_l}")
            except ValueError:
                 if val_r != val_l:
                     mismatches.append(f"[{key}] {attr} string mismatch")

    if mismatches:
        print(f"\n[FAIL] Found {len(mismatches)} symmetry issues:\n")
        for m in mismatches:
            print(m)
            print("-" * 40)
    else:
        print("\n[SUCCESS] Perfect symmetry maintained (tolerance 1e-5).")

def apply_symmetry(xml_path):
    tree, muscles_r, muscles_l = get_muscles(xml_path)
    if tree is None: return

    updated_count = 0
    target_attrs = ['biasprm', 'gainprm', 'lengthrange']
    
    for key, r_elem in muscles_r.items():
        if key not in muscles_l:
            print(f"Warning: No matching left muscle for {key}_r")
            continue
            
        l_elem = muscles_l[key]
        
        # Copy attributes from Right to Left
        changed = False
        for attr in target_attrs:
            val_r = r_elem.get(attr)
            if val_r is not None:
                if l_elem.get(attr) != val_r:
                    l_elem.set(attr, val_r)
                    changed = True
        
        if changed:
            updated_count += 1

    if updated_count > 0:
        print(f"Updated {updated_count} muscles (Right -> Left).")
        tree.write(xml_path, encoding='utf-8', xml_declaration=False)
        print(f"Saved changes to {xml_path}")
    else:
        print("No changes needed (already symmetric).")

def main():
    parser = argparse.ArgumentParser(description="Manage symmetry in MuJoCo XML muscle files. Automatically checks and applies Right->Left symmetry.")
    parser.add_argument('--files', nargs='*', help="Specific files to process. If empty, processes default rigid/compliant files.")
    
    args = parser.parse_args()
    
    # Determine repo root (parent of 'scripts')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    default_files = [
        os.path.join(repo_root, "myosim_convert", "myo_sim", "leg", "assets", "myolegs_muscle_rigid.xml"),
        os.path.join(repo_root, "myosim_convert", "myo_sim", "leg", "assets", "myolegs_muscle_compliant.xml")
    ]
    
    files = args.files if args.files else default_files
    
    for f in files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(f)}")
        print(f"{'='*60}")
        
        # 1. Check
        print(">>> Checking Symmetry...")
        check_symmetry(f)
        
        # 2. Apply
        print("\n>>> Applying Symmetry (Right -> Left)...")
        apply_symmetry(f)

if __name__ == "__main__":
    main()
