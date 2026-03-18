BRANCHES = {
    "ecal": {
        "simhits_pileup": {
            "scalars": [],
            "vectors": {
                "x": "EcalSimHits_pileup/EcalSimHits_pileup.x_",
                "y": "EcalSimHits_pileup/EcalSimHits_pileup.y_",
                "z": "EcalSimHits_pileup/EcalSimHits_pileup.z_",
                "energy": "EcalSimHits_pileup/EcalSimHits_pileup.edep_",
            },
        },
        "rechits_overlay": {
            "scalars": [],
            "vectors": {
                "x": "EcalRecHits_overlay/EcalRecHits_overlay.xpos_",
                "y": "EcalRecHits_overlay/EcalRecHits_overlay.ypos_",
                "z": "EcalRecHits_overlay/EcalRecHits_overlay.zpos_",
                "energy": "EcalRecHits_overlay/EcalRecHits_overlay.energy_",
            },
        },
    }
}


def get_collection(detector: str, collection: str) -> dict:
    return BRANCHES[detector][collection]


def get_vector_branches(detector: str, collection: str) -> dict:
    return BRANCHES[detector][collection]["vectors"]


def get_scalar_branches(detector: str, collection: str) -> list:
    return BRANCHES[detector][collection]["scalars"]


def get_all_branch_names(detector: str, collection: str) -> list:
    cfg = BRANCHES[detector][collection]
    return cfg["scalars"] + list(cfg["vectors"].values())