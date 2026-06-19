
action: Action = Action.NOTHING
if result.decision == "pet":
    action = Action.PET
elif result.decision == "slap_left":
    action = Action.SLAP_LEFT
elif result.decision == "slap_right":
    action = Action.SLAP_RIGHT