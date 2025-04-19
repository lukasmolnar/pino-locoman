
# Default OCP arguments for each dynamics method
OCP_ARGS = {
    "centroidal_vel": {
        "include_base": False,  # whether base velocity is part of the input
    },
    "centroidal_acc": {
        "include_base": False,  # whether base acceleration is part of the input
    },
    "whole_body_acc": {
        "include_base": False,  # whether base acceleration is part of the input
    },
    "whole_body_aba": {
        # the input just contains joint torques
    },
    "whole_body_rnea": {
        "tau_nodes": 3,  # after this many nodes, joint torques are removed from the input
        "include_acc": True,  # whether to include accelerations in the input (necessary for Fatrop due to structure detection!)
    }
}