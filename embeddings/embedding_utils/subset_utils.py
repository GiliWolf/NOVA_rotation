import sys
import os
sys.path.insert(0, os.getenv("HOME"))


def _extract_mutual_params(mutual_param):
        if isinstance(mutual_param, str):
                    mutual_param_c1 = mutual_param
                    mutual_param_c2 = mutual_param
                    return mutual_param, mutual_param
        elif isinstance(mutual_param, list):
                if len(mutual_param) == 1:
                    return mutual_param[0], mutual_param[0]
                else:
                    return mutual_param[0], mutual_param[1]
        else:
            raise ValueError("[get subset: extract_subset] Mutual param is not a valid option.")
