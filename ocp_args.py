
# Default OCP arguments for each dynamics method
OCP_ARGS = {
    "centroidal_vel": {
        "include_base": True,  # whether base velocity is part of the input
    },
    "centroidal_acc": {
        "include_base": True,  # whether base acceleration is part of the input
    },
    "whole_body_acc": {
        "include_base": True,  # whether base acceleration is part of the input
    },
    "whole_body_aba": {
        # the input just contains joint torques
    },
    "whole_body_rnea": {
        # the input contains base + joint accelerations, and joint torques
        "tau_nodes": 3,  # after this many nodes, joint torques are removed from the input
    }
}