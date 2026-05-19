# mldmx/src/mldmx/io/root_reader.py

from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import uproot

from mldmx.io.branches import get_all_branch_names, get_vector_branches

'''
This module provides utilities for reading ROOT files using uproot with awkward arrays. 
It defines a RootSource dataclass to specify the file and tree, and a read_branches function to read specified branches into awkward arrays.
'''

@dataclass(frozen=True)
class RootSource:
    path: str
    tree_name: str = "LDMX_Events"


def read_branches(source: RootSource, branch_names, entry_start=0, entry_stop=None):
    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        arrays = tree.arrays(branch_names, entry_start=entry_start, entry_stop=entry_stop, library="ak")
    return arrays


def _read_tree_arrays(tree, branch_names, entry_start=0, entry_stop=None, step_size=None):
    if step_size is None:
        yield tree.arrays(
            branch_names,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
        )
        return

    yield from tree.iterate(
        branch_names,
        entry_start=entry_start,
        entry_stop=entry_stop,
        library="ak",
        step_size=step_size,
    )


def _missing_branches(tree, branch_names):
    available_branches = set(tree.keys())
    return [
        branch_name
        for branch_name in branch_names
        if branch_name not in available_branches
    ]


def has_branches(source: RootSource, branch_names) -> bool:
    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        available_branches = set(tree.keys())
    return all(branch_name in available_branches for branch_name in branch_names)


def select_collection(source: RootSource, detector: str, collections):
    for collection in collections:
        branch_names = get_all_branch_names(detector, collection)
        if has_branches(source, branch_names):
            vectors = get_vector_branches(detector, collection)
            return collection, vectors, branch_names

    raise KeyError(
        f"Could not find any supported {detector} branch layout. "
        f"Tried: {list(collections)}"
    )


def read_events(root_path, max_events=10):
    """
    Read ECal hit-level information from an LDMX ROOT file and return a list
    of per-event dictionaries with plain Python lists.

    Each event has the form:
        {
            "x":      list[float] [N_hits],
            "y":      list[float] [N_hits],
            "z":      list[float] [N_hits],
            "energy": list[float] [N_hits],
        }

    The function first tries the overlay rechit branch layout used in events.root.
    If that fails, it falls back to the simhit layout used in pileup.root.
    """

    root_path = Path(root_path)
    source = RootSource(path=str(root_path), tree_name="LDMX_Events")

    _branch_type, vectors, branch_names = select_collection(
        source,
        detector="ecal",
        collections=["rechits_overlay", "simhits_pileup"],
    )
    arrays = read_branches(
        source,
        branch_names=branch_names,
        entry_start=0,
        entry_stop=max_events,
    )

    x_branch = vectors["x"]
    y_branch = vectors["y"]
    z_branch = vectors["z"]
    energy_branch = vectors["energy"]

    num_events = len(arrays[x_branch])
    events = []

    for i in range(num_events):
        event = {
            "x": [float(v) for v in ak.to_list(arrays[x_branch][i])],
            "y": [float(v) for v in ak.to_list(arrays[y_branch][i])],
            "z": [float(v) for v in ak.to_list(arrays[z_branch][i])],
            "energy": [float(v) for v in ak.to_list(arrays[energy_branch][i])],
        }
        events.append(event)

    return events


def read_ecal_rechits_with_truth(root_path, max_events=10):
    """
    Read overlay ECal RecHits and align their noise flags plus sim-hit
    contribution truth by detector hit ID.

    Each event has the form:
        {
            "hit_id": list[int] [N_hits],
            "x": list[float] [N_hits],
            "y": list[float] [N_hits],
            "z": list[float] [N_hits],
            "energy": list[float] [N_hits],
            "noise_flag": list[bool] [N_hits],
            "track_id_contribs": list[list[int]] [N_hits][variable],
            "edep_contribs": list[list[float]] [N_hits][variable],
            "origin_id_contribs": list[list[int]] [N_hits][variable],
            "n_contribs": list[int] [N_hits],
        }   

    """

    root_path = Path(root_path)
    source = RootSource(path=str(root_path), tree_name="LDMX_Events")

    rechit_vectors = get_vector_branches("ecal", "rechits_overlay")
    simhit_vectors = get_vector_branches("ecal", "simhits_overlay")
    branch_names = list(rechit_vectors.values()) + list(simhit_vectors.values())

    missing_branches = []
    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        available_branches = set(tree.keys())
        for branch_name in branch_names:
            if branch_name not in available_branches:
                missing_branches.append(branch_name)

    if missing_branches:
        raise KeyError(f"Missing required overlay truth branches: {missing_branches}")

    arrays = read_branches(
        source,
        branch_names=branch_names,
        entry_start=0,
        entry_stop=max_events,
    )

    num_events = len(arrays[rechit_vectors["id"]])
    events = []

    for iev in range(num_events):
        simhit_ids = ak.to_list(arrays[simhit_vectors["id"]][iev])
        simhit_truth_by_id = {}
        for idx, hit_id in enumerate(simhit_ids):
            hit_id = int(hit_id)
            simhit_truth_by_id[hit_id] = {
                "track_id_contribs": [int(v) for v in ak.to_list(arrays[simhit_vectors["track_id_contribs"]][iev][idx])],
                "edep_contribs": [float(v) for v in ak.to_list(arrays[simhit_vectors["edep_contribs"]][iev][idx])],
                "origin_id_contribs": [int(v) for v in ak.to_list(arrays[simhit_vectors["origin_id_contribs"]][iev][idx])],
                "n_contribs": int(arrays[simhit_vectors["n_contribs"]][iev][idx]),
            }

        hit_ids = [int(v) for v in ak.to_list(arrays[rechit_vectors["id"]][iev])]
        event = {
            "hit_id": hit_ids,
            "x": [float(v) for v in ak.to_list(arrays[rechit_vectors["x"]][iev])],
            "y": [float(v) for v in ak.to_list(arrays[rechit_vectors["y"]][iev])],
            "z": [float(v) for v in ak.to_list(arrays[rechit_vectors["z"]][iev])],
            "energy": [float(v) for v in ak.to_list(arrays[rechit_vectors["energy"]][iev])],
            "noise_flag": [bool(v) for v in ak.to_list(arrays[rechit_vectors["noise_flag"]][iev])],
            "track_id_contribs": [],
            "edep_contribs": [],
            "origin_id_contribs": [],
            "n_contribs": [],
        }

        for hit_id in hit_ids:
            truth = simhit_truth_by_id.get(
                hit_id,
                {
                    "track_id_contribs": [],
                    "edep_contribs": [],
                    "origin_id_contribs": [],
                    "n_contribs": 0,
                },
            )
            event["track_id_contribs"].append(truth["track_id_contribs"])
            event["edep_contribs"].append(truth["edep_contribs"])
            event["origin_id_contribs"].append(truth["origin_id_contribs"])
            event["n_contribs"].append(truth["n_contribs"])

        events.append(event)

    return events


def _build_ecal_tpad_context_events(arrays, rechit_vectors, simhit_vectors, tpad_vectors):
    num_events = len(arrays[rechit_vectors["id"]])

    for iev in range(num_events):
        simhit_ids = ak.to_list(arrays[simhit_vectors["id"]][iev])
        simhit_truth_by_id = {}
        for idx, hit_id in enumerate(simhit_ids):
            hit_id = int(hit_id)
            simhit_truth_by_id[hit_id] = {
                "track_id_contribs": [
                    int(v)
                    for v in ak.to_list(
                        arrays[simhit_vectors["track_id_contribs"]][iev][idx]
                    )
                ],
                "edep_contribs": [
                    float(v)
                    for v in ak.to_list(
                        arrays[simhit_vectors["edep_contribs"]][iev][idx]
                    )
                ],
                "origin_id_contribs": [
                    int(v)
                    for v in ak.to_list(
                        arrays[simhit_vectors["origin_id_contribs"]][iev][idx]
                    )
                ],
                "n_contribs": int(arrays[simhit_vectors["n_contribs"]][iev][idx]),
            }

        hit_ids = [int(v) for v in ak.to_list(arrays[rechit_vectors["id"]][iev])]
        event = {
            "hit_id": hit_ids,
            "x": [float(v) for v in ak.to_list(arrays[rechit_vectors["x"]][iev])],
            "y": [float(v) for v in ak.to_list(arrays[rechit_vectors["y"]][iev])],
            "z": [float(v) for v in ak.to_list(arrays[rechit_vectors["z"]][iev])],
            "energy": [
                float(v)
                for v in ak.to_list(arrays[rechit_vectors["energy"]][iev])
            ],
            "noise_flag": [
                bool(v)
                for v in ak.to_list(arrays[rechit_vectors["noise_flag"]][iev])
            ],
            "track_id_contribs": [],
            "edep_contribs": [],
            "origin_id_contribs": [],
            "n_contribs": [],
            "trigger_pad_tracks": {
                "centroid": [
                    float(v)
                    for v in ak.to_list(arrays[tpad_vectors["centroid"]][iev])
                ],
                "pe": [float(v) for v in ak.to_list(arrays[tpad_vectors["pe"]][iev])],
            },
        }

        for hit_id in hit_ids:
            truth = simhit_truth_by_id.get(
                hit_id,
                {
                    "track_id_contribs": [],
                    "edep_contribs": [],
                    "origin_id_contribs": [],
                    "n_contribs": 0,
                },
            )
            event["track_id_contribs"].append(truth["track_id_contribs"])
            event["edep_contribs"].append(truth["edep_contribs"])
            event["origin_id_contribs"].append(truth["origin_id_contribs"])
            event["n_contribs"].append(truth["n_contribs"])

        yield event


def read_trigger_pad_tracks(root_path, max_events=10):
    """
    Read TriggerPadTracks overlay context information.

    Each event has the form:
        {
            "centroid": list[float] [N_tracks],
            "pe": list[float] [N_tracks],
        }

    Only centroid_ and pe_ are read. The centroid_ field is the useful 1D
    coordinate for this detector context in the current prototype.
    """

    root_path = Path(root_path)
    source = RootSource(path=str(root_path), tree_name="LDMX_Events")

    vectors = get_vector_branches("trigger_pad_tracks", "overlay")
    branch_names = list(vectors.values())

    missing_branches = []
    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        available_branches = set(tree.keys())
        for branch_name in branch_names:
            if branch_name not in available_branches:
                missing_branches.append(branch_name)

    if missing_branches:
        raise KeyError(f"Missing required TriggerPadTracks branches: {missing_branches}")

    arrays = read_branches(
        source,
        branch_names=branch_names,
        entry_start=0,
        entry_stop=max_events,
    )

    num_events = len(arrays[vectors["centroid"]])
    events = []
    for iev in range(num_events):
        events.append(
            {
                "centroid": [float(v) for v in ak.to_list(arrays[vectors["centroid"]][iev])],
                "pe": [float(v) for v in ak.to_list(arrays[vectors["pe"]][iev])],
            }
        )

    return events


def read_ecal_rechits_with_truth_and_triggerpad_context(root_path, max_events=10):
    """
    Read labelled ECal RecHits and attach TriggerPadTracks overlay context.

    Existing ECal-only readers remain unchanged; new scripts opt into this
    combined event shape explicitly.
    """

    return [
        event
        for _entry, event in iter_ecal_rechits_with_truth_and_triggerpad_context(
            root_path,
            max_events=max_events,
        )
    ]


def iter_ecal_rechits_with_truth_and_triggerpad_context(
    root_path,
    max_events=10,
    step_size=None,
):
    """
    Yield labelled ECal RecHits with TriggerPadTracks context from one ROOT file.

    This reads all required ECal, truth, and TriggerPadTracks branches through one
    tree handle. Passing step_size enables chunked uproot reads, which lets callers
    show progress between chunks and caps peak memory for larger samples.
    """

    root_path = Path(root_path)
    source = RootSource(path=str(root_path), tree_name="LDMX_Events")
    rechit_vectors = get_vector_branches("ecal", "rechits_overlay")
    simhit_vectors = get_vector_branches("ecal", "simhits_overlay")
    tpad_vectors = get_vector_branches("trigger_pad_tracks", "overlay")
    branch_names = (
        list(rechit_vectors.values())
        + list(simhit_vectors.values())
        + list(tpad_vectors.values())
    )

    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        missing_branches = _missing_branches(tree, branch_names)
        if missing_branches:
            raise KeyError(f"Missing required ECal/TriggerPadTracks branches: {missing_branches}")

        entry_stop = max_events
        if max_events is not None:
            entry_stop = min(int(max_events), int(tree.num_entries))

        entry_offset = 0
        for arrays in _read_tree_arrays(
            tree,
            branch_names=branch_names,
            entry_start=0,
            entry_stop=entry_stop,
            step_size=step_size,
        ):
            for local_offset, event in enumerate(
                _build_ecal_tpad_context_events(
                    arrays,
                    rechit_vectors=rechit_vectors,
                    simhit_vectors=simhit_vectors,
                    tpad_vectors=tpad_vectors,
                )
            ):
                yield entry_offset + local_offset, event
            entry_offset += len(arrays[rechit_vectors["id"]])
