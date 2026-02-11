"""
Annotation module for adding fingering to MEI files.

Uses either a trained PPO agent or the linear (Dijkstra) agent to
compute fingerings, then writes them back into MEI XML files.
"""

import os
import time
import copy
import xml.etree.ElementTree as ET
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import counterpoint.envs  # Register envs

from counterpoint.scores.mei import parse_mei_file, MEIScoreGenerator


# MEI namespace
MEI_NS = "http://www.music-encoding.org/ns/mei"


def _collect_fingerings_linear(env, rules: str = "jacobs") -> List[Tuple[str, int]]:
    """
    Run the linear agent through the env and collect fingerings.
    
    Returns:
        List of (finger_number) for each note in order
    """
    from counterpoint.linear.agent import LinearAgent
    
    agent = LinearAgent()
    
    obs, _ = env.reset()
    terminated = False
    truncated = False
    fingerings = []
    
    while not (terminated or truncated):
        action = agent.solve(env)
        
        # Extract which finger was pressed
        fingers = action["fingers"]
        pressed = [i + 1 for i, f in enumerate(fingers) if f == 1]
        finger = pressed[0] if pressed else 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if note was successfully played (step advanced)
        # We track based on the env's current step
        fingerings.append(finger)
    
    return fingerings


def _collect_fingerings_ppo(env, checkpoint_path: str) -> List[int]:
    """
    Run the PPO agent on the env and collect fingerings.
    
    Returns:
        List of finger numbers for each note in order
    """
    import torch
    from counterpoint.train import load_config, get_models
    from counterpoint.policies import FlattenActionWrapper
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.envs.wrappers.torch import wrap_env
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config()
    
    # Wrap env for SKRL
    flat_env = FlattenActionWrapper(env)
    wrapped_env = wrap_env(flat_env)
    
    # Build agent
    policy_type = config.get("policy", {}).get("type", "simple")
    models = get_models(policy_type, wrapped_env.observation_space, 
                       wrapped_env.action_space, device)
    
    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    exp_cfg = config["experiment"]
    cfg_agent["experiment"]["directory"] = exp_cfg["directory"]
    
    agent = PPO(
        models=models,
        memory=None,
        cfg=cfg_agent,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=device,
    )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    
    # Run inference
    states, _ = wrapped_env.reset()
    fingerings = []
    
    while True:
        with torch.no_grad():
            actions = agent.act(states, timestep=0, timesteps=0)[0]
        
        states, rewards, terminated, truncated, infos = wrapped_env.step(actions)
        
        # Extract finger from action (first 5 values are finger presses)
        action_np = actions.cpu().numpy().flatten()
        finger_presses = action_np[:5]
        pressed = [i + 1 for i, f in enumerate(finger_presses) if f > 0.5]
        finger = pressed[0] if pressed else 1
        fingerings.append(finger)
        
        if terminated.any() or truncated.any():
            break
    
    return fingerings


def write_fingerings_to_mei(
    mei_path: str, 
    output_path: str,
    note_ids: List[str],
    fingerings: List[int],
    staff: int = 1,
    strip_all_existing: bool = True,
):
    """
    Write fingering annotations into an MEI file.
    
    Adds <fing> elements to measures, linking each to a note via startid.
    
    Args:
        mei_path: Path to the source MEI file
        output_path: Path to write the annotated MEI file
        note_ids: List of note xml:id strings
        fingerings: List of finger numbers (1-5)
        staff: Staff number to annotate
        strip_all_existing: If True, remove ALL existing <fing> elements first.
            If False, only remove <fing> elements for the target staff.
    """
    # Register MEI namespace
    ET.register_namespace("", MEI_NS)
    ET.register_namespace("xml", "http://www.w3.org/XML/1998/namespace")
    
    tree = ET.parse(mei_path)
    root = tree.getroot()
    
    # Remove existing <fing> elements
    fings_to_remove = []
    for fing in root.iter(f"{{{MEI_NS}}}fing"):
        if strip_all_existing:
            fings_to_remove.append(fing)
        else:
            fing_staff = fing.get("staff")
            if fing_staff is None or fing_staff == str(staff):
                fings_to_remove.append(fing)
    
    for fing in fings_to_remove:
        parent = _find_parent(root, fing)
        if parent is not None:
            parent.remove(fing)
    
    if fings_to_remove:
        print(f"  Stripped {len(fings_to_remove)} existing <fing> elements")
    
    # Build a mapping: note_id -> finger
    id_to_finger = {}
    for note_id, finger in zip(note_ids, fingerings):
        if note_id not in id_to_finger:
            id_to_finger[note_id] = finger
    
    # Find all measures and add <fing> elements
    xml_ns = "http://www.w3.org/XML/1998/namespace"
    fings_added = 0
    for measure in root.iter(f"{{{MEI_NS}}}measure"):
        # Find all notes in this measure for the target staff
        for staff_elem in measure.findall(f"{{{MEI_NS}}}staff"):
            staff_n = staff_elem.get("n")
            if staff_n != str(staff):
                continue
            
            for note in staff_elem.iter(f"{{{MEI_NS}}}note"):
                note_id = note.get(f"{{{xml_ns}}}id")
                if note_id is None:
                    # Try other attribute forms
                    for attr_name in note.attrib:
                        if attr_name.endswith("}id") or attr_name == "xml:id":
                            note_id = note.attrib[attr_name]
                            break
                
                if note_id and note_id in id_to_finger:
                    fing_elem = ET.SubElement(measure, f"{{{MEI_NS}}}fing")
                    fing_elem.set("staff", str(staff))
                    fing_elem.set("startid", f"#{note_id}")
                    fing_elem.text = str(id_to_finger[note_id])
                    fings_added += 1
    
    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tree.write(output_path, xml_declaration=True, encoding="UTF-8")
    print(f"  Written annotated MEI to: {output_path} ({fings_added} fingerings added)")


def _find_parent(root, target):
    """Find the parent element of a target element in the XML tree."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


def annotate_mei_with_linear(
    input_path: str,
    output_dir: str,
    rules: str = "jacobs",
    staff: int = 1,
    both: bool = False,
):
    """
    Annotate MEI file(s) with fingering using the linear (Dijkstra) agent.
    
    Args:
        input_path: Path to MEI file or directory of MEI files
        output_dir: Directory to write annotated files
        rules: Rule set to use ("jacobs" or "parncutt")
        staff: Staff to annotate (1=treble, 2=bass). Ignored when both=True.
        both: Annotate both staves (RH then LH)
    """
    staves = [1, 2] if both else [staff]
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        mei_files = [input_path]
    elif input_path.is_dir():
        mei_files = sorted(input_path.glob("*.mei"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if not mei_files:
        raise FileNotFoundError(f"No .mei files found in {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for mei_file in mei_files:
        print(f"\nAnnotating: {mei_file.name}")
        output_path = os.path.join(output_dir, mei_file.name)
        
        for staff_idx, s in enumerate(staves):
            hand_label = "RH" if s == 1 else "LH"
            print(f"  Staff {s} ({hand_label})...")
            
            # Parse to get note IDs
            _, note_ids = parse_mei_file(str(mei_file), staff=s)
            
            # Create env with this specific file
            env = gym.make(
                "Piano-v0",
                score_generator_type="mei",
                mei_path=str(mei_file),
                mei_loop=False,
                hand=s,
            )
            
            # Run linear agent
            fingerings = _collect_fingerings_linear(env, rules=rules)
            env.close()
            
            # Match fingerings to note IDs
            score_len = len(note_ids)
            if len(fingerings) > score_len:
                fingerings = fingerings[:score_len]
            elif len(fingerings) < score_len:
                fingerings.extend([1] * (score_len - len(fingerings)))
            
            # For both mode: first staff reads from original, second reads from output
            # strip_all_existing=True only for the first staff in both mode
            source = str(mei_file) if staff_idx == 0 else output_path
            write_fingerings_to_mei(
                source, output_path, note_ids, fingerings,
                staff=s,
                strip_all_existing=(staff_idx == 0),
            )
            
            print(f"  Notes: {score_len}, Fingerings applied: {len(fingerings)}")


def annotate_mei_with_ppo(
    input_path: str,
    output_dir: str,
    checkpoint_path: str,
    staff: int = 1,
    both: bool = False,
):
    """
    Annotate MEI file(s) with fingering using a trained PPO agent.
    
    Args:
        input_path: Path to MEI file or directory of MEI files
        output_dir: Directory to write annotated files
        checkpoint_path: Path to PPO model checkpoint
        staff: Staff to annotate (1=treble, 2=bass). Ignored when both=True.
        both: Annotate both staves (RH then LH)
    """
    staves = [1, 2] if both else [staff]
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        mei_files = [input_path]
    elif input_path.is_dir():
        mei_files = sorted(input_path.glob("*.mei"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if not mei_files:
        raise FileNotFoundError(f"No .mei files found in {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for mei_file in mei_files:
        print(f"\nAnnotating: {mei_file.name}")
        output_path = os.path.join(output_dir, mei_file.name)
        
        for staff_idx, s in enumerate(staves):
            hand_label = "RH" if s == 1 else "LH"
            print(f"  Staff {s} ({hand_label})...")
            
            # Parse to get note IDs
            _, note_ids = parse_mei_file(str(mei_file), staff=s)
            
            # Create env with this specific file
            env = gym.make(
                "Piano-v0",
                score_generator_type="mei",
                mei_path=str(mei_file),
                mei_loop=False,
                hand=s,
            )
            
            # Run PPO agent
            fingerings = _collect_fingerings_ppo(env, checkpoint_path)
            env.close()
            
            # Match fingerings to note IDs
            score_len = len(note_ids)
            if len(fingerings) > score_len:
                fingerings = fingerings[:score_len]
            elif len(fingerings) < score_len:
                fingerings.extend([1] * (score_len - len(fingerings)))
            
            # For both mode: first staff reads from original, second reads from output
            source = str(mei_file) if staff_idx == 0 else output_path
            write_fingerings_to_mei(
                source, output_path, note_ids, fingerings,
                staff=s,
                strip_all_existing=(staff_idx == 0),
            )
            
            print(f"  Notes: {score_len}, Fingerings applied: {len(fingerings)}")

