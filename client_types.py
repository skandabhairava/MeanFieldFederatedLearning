from enum import StrEnum

class ClientTypes(StrEnum):
    NORMAL = "normal"
    SUBTLE = "subtle"
    ALIE = "alie" ## DEPRECATED
    IPM = "ipm"
    BACKDOOR_0 = "backdoor_0"
    BACKDOOR_0_ALL = "backdoor_0_all"
    LABEL_SWITCH = "label_switch"
    RANDOM = "random"
    SIGN_FLIP = "sign_flip"
    MODEL_REPLACE_POS = "model_replace_pos"

    LATE_SUBTLE = "late_subtle"
    LATE_ALIE = "late_alie" ## DEPRECATED
    LATE_IPM = "late_ipm"
    LATE_BACKDOOR_0 = "late_backdoor_0"
    LATE_LABEL_SWITCH = "late_label_switch"
