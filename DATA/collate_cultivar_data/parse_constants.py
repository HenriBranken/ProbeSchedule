def get_val_from_assignment(keyword, master_list, convert_to="string"):
    for i, assignment in enumerate(master_list):
        if keyword in master_list[i]:
            val = assignment.split(keyword, 1)[1][1:]
            break
    else:
        val = 99999

    if (convert_to == "int") and (val != 99999):
        return int(val)
    elif (convert_to == "float") and (val != 99999):
        return float(val)
    elif (convert_to == "string") and (val != 99999):
        return str(val)
    elif (convert_to == "bool") and (val != 99999):
        if (val == "True") and (val != 99999):
            return True
        elif (val == "False") and (val != 99999):
            return False
        else:
            raise KeyError
    else:
        raise KeyError


with open("constants.txt", "r") as f:
    constants_list = [x.rstrip("\n") for x in list(f) if x != "\n"]

RAIN_THRESHOLD = get_val_from_assignment("RAIN_THRESHOLD", constants_list,
                                         "float")


ETO_MAX = get_val_from_assignment("ETO_MAX", constants_list, "float")


KCP_MAX = get_val_from_assignment("KCP_MAX", constants_list, "float")


BEGINNING_MONTH = get_val_from_assignment("BEGINNING_MONTH", constants_list,
                                          "int")


ETCP_PERC_DEVIATION = get_val_from_assignment("ETCP_PERC_DEVIATION",
                                              constants_list, "float")


KCP_PERC_DEVIATION = get_val_from_assignment("KCP_PERC_DEVIATION",
                                             constants_list, "float")


start_date = get_val_from_assignment("start_date", constants_list, "string")


T_base = get_val_from_assignment("T_base", constants_list, "float")


pol_degree = get_val_from_assignment("pol_degree", constants_list, "int")


ALLOWED_TAIL_DEVIATION = get_val_from_assignment("ALLOWED_TAIL_DEVIATION",
                                                 constants_list, "float")

delta_x = get_val_from_assignment("delta_x", constants_list, "int")


x_limits_left = get_val_from_assignment("x_limits_left", constants_list, "int")
x_limits_right = get_val_from_assignment("x_limits_right", constants_list,
                                         "int")
x_limits = [x_limits_left, x_limits_right]


CULTIVAR = get_val_from_assignment("CULTIVAR", constants_list, "string")


WEEKLY_BINNED_VERSION = get_val_from_assignment("WEEKLY_BINNED_VERSION",
                                                constants_list, "bool")


mode = get_val_from_assignment("mode", constants_list, "string")


constants_dict = {"RAIN_THRESHOLD": RAIN_THRESHOLD,
                  "ETO_MAX": ETO_MAX,
                  "KCP_MAX": KCP_MAX,
                  "BEGINNING_MONTH": BEGINNING_MONTH,
                  "ETCP_PERC_DEVIATION": ETCP_PERC_DEVIATION,
                  "KCP_PERC_DEVIATION": KCP_PERC_DEVIATION,
                  "start_date": start_date,
                  "T_base": T_base,
                  "pol_degree": pol_degree,
                  "ALLOWED_TAIL_DEVIATION": ALLOWED_TAIL_DEVIATION,
                  "delta_x": delta_x,
                  "x_limits": x_limits,
                  "CULTIVAR": CULTIVAR,
                  "WEEKLY_BINNED_VERSION": WEEKLY_BINNED_VERSION,
                  "mode": mode}

"""
print("RAIN_THRESHOLD = {:.1f}.  "
      "type(RAIN_THRESHOLD) = {:s}.".format(RAIN_THRESHOLD,
                                            str(type(RAIN_THRESHOLD))))
print("ETO_MAX = {:.1f}.  type(ETO_MAX) = {:s}.".format(ETO_MAX,
                                                        str(type(ETO_MAX))))
print("KCP_MAX = {:.1f}.  type(KCP_MAX) = {:s}.".format(KCP_MAX,
                                                        str(type(KCP_MAX))))
print("BEGINNING_MONTH = {:.0f}.  "
      "type(BEGINNING_MONTH) = {:s}.".format(BEGINNING_MONTH,
                                             str(type(BEGINNING_MONTH))))
print("ETCP_PERC_DEVIATION= {:.2f}.  type(ETCP_PERC_DEVIATION) = "
      "{:s}.".format(ETCP_PERC_DEVIATION, str(type(ETCP_PERC_DEVIATION))))
print("KCP_PERC_DEVIATION = {:.2f}.  type(KCP_PERC_DEVIATION) = "
      "{:s}.".format(KCP_PERC_DEVIATION, str(type(KCP_PERC_DEVIATION))))
print("start_date = {:s}.  type(start_date) = "
      "{:s}.".format(start_date, str(type(start_date))))  
print("T_base = {:.1f}.  type(T_base) = {:s}.".format(T_base,
                                                      str(type(T_base))))
print("pol_degree = {:.1f}.  type(pol_degree) = "
      "{:s}.".format(pol_degree, str(type(pol_degree))))
print("ALLOWED_TAIL_DEVIATION = {:.2f}.  type(ALLOWED_TAIL_DEVAITION) = "
      "{:s}.".format(ALLOWED_TAIL_DEVIATION,
                     str(type(ALLOWED_TAIL_DEVIATION))))
print("delta_x = {:.0f}.  type(delta_x) = {:s}.".format(delta_x,
                                                        str(type(delta_x))))
print("x_limits = {}.  type(x_limits) = {:s}.".format(x_limits,
                                                      str(type(x_limits))))
print("CULTIVAR = {}.  type(CULTIVAR) = {:s}.".format(CULTIVAR,
                                                      str(type(CULTIVAR))))  
print("WEEKLY_BINNED_VERSION = {}.  type(WEEKLY_BINNED_VERSION) = "
      "{:s}.".format(WEEKLY_BINNED_VERSION, str(type(WEEKLY_BINNED_VERSION))))
print("mode = {:s}.  type(mode) = {:s}.".format(mode, str(type(mode))))                                                                                                                                                                                                                                                                                                                     
"""
