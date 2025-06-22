import sys
import os
sys.path.insert(0, os.getenv("HOME"))


def _extract_mutual_params(mutual_param):
        if isinstance(mutual_param, list):
                if len(mutual_param) == 1:
                    return mutual_param[0], mutual_param[0]
                else:
                    return mutual_param[0], mutual_param[1]
        else:
            raise ValueError("[get subset: extract_subset] Mutual param is not a valid option.")

def _get_subset_params(data_config):
        mutual_param_c1, mutual_param_c2 = _extract_mutual_params(data_config[data_config.MUTUAL_ATTR])
        compare_by_attr_list:list = data_config[data_config.COMPARE_BY_ATTR]
        compare_param_c1 = compare_by_attr_list[0]
        compare_param_c2 = compare_by_attr_list[1]