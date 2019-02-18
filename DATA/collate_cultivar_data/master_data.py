import numpy as np
import pandas as pd

IMPUTED_ETO = "Imputed eto"

# Long-term eto data for Koue Bokkeveld
calendar_week = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                          35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                          45, 46, 47, 48, 49,  1,  2,  3,  4,  5,
                          6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 24])

kbv_eto = np.array([2.30, 2.30, 2.30, 2.30, 2.30, 2.40, 2.50, 2.65, 2.80, 3.10,
                    3.40, 3.65, 4.00, 4.40, 4.80, 5.30, 5.80, 6.30, 6.70, 7.10,
                    7.60, 8.00, 8.30, 8.60, 8.80, 8.90, 8.90, 8.90, 8.80, 8.70,
                    8.50, 8.30, 8.00, 7.50, 7.00, 6.50, 5.80, 5.20, 4.70, 4.30,
                    3.70, 3.40, 3.10, 2.80, 2.50, 2.45, 2.40, 2.35, 2.30])

df_kbv = pd.DataFrame(data=kbv_eto, index=calendar_week, columns=["kbv_eto"])

df_kbv.index.name = "calendar_week"


# Master Crop Coefficients and kcp flagging function
calendar_month = np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
# penman_kcp = np.array([0.10, 0.30, 0.40, 0.60, 0.73, 0.88, 0.95, 0.95, 0.95, 0.70, 0.40, 0.20])
row_4 = np.array([0.61, 0.73, 0.88, 0.95, 0.95, 0.95, 0.95, 0.90, 0.80, 0.40, 0.30, 0.10])

accepted_kcp_norm = pd.DataFrame(index=calendar_month, data=row_4, columns=["norm_kcp"])
accepted_kcp_norm.index.name = "calendar_month"
