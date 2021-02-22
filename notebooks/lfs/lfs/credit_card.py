from freedive.labeling import labeling_function

@labeling_function()
def lf01(sample):
    if sample['PAY_0'] <= 1.5:
        if sample['PAY_0'] <= 0.5:
            if sample['PAY_AMT2'] <= 1500.5:
                # N. of samples 7870 ([6337. 1533.])
                return 0
            else:  # if sample['PAY_AMT2'] > 1500.5
                if sample['PAY_AMT5'] <= 1930.5:
                    # N. of samples 5604 ([4827.  777.])
                    return 0
                else:  # if sample['PAY_AMT5'] > 1930.5
                    # N. of samples 9708 ([8811.  897.])
                    return 0
        else:  # if sample['PAY_0'] > 0.5
            # N. of samples 3688 ([2436. 1252.])
            return 0
    else:  # if sample['PAY_0'] > 1.5
        # N. of samples 3130 ([ 953. 2177.])
        return 1

@labeling_function()
def lf02(sample):
    if sample['PAY_3'] <= 1.5:
        if sample['PAY_AMT1'] <= 21.5:
            # N. of samples 4301 ([3026. 1275.])
            return 0
        else:  # if sample['PAY_AMT1'] > 21.5
            if sample['PAY_AMT6'] <= 1502.5:
                if sample['LIMIT_BAL'] <= 75000.0:
                    # N. of samples 4265 ([3305.  960.])
                    return 0
                else:  # if sample['LIMIT_BAL'] > 75000.0
                    # N. of samples 5263 ([4504.  759.])
                    return 0
            else:  # if sample['PAY_AMT6'] > 1502.5
                # N. of samples 11962 ([10521.  1441.])
                return 0
    else:  # if sample['PAY_3'] > 1.5
        # N. of samples 4209 ([2008. 2201.])
        return 1

@labeling_function()
def lf03(sample):
    if sample['PAY_AMT4'] <= 0.5:
        if sample['LIMIT_BAL'] <= 135000.0:
            # N. of samples 3160 ([1985. 1175.])
            return 0
        else:  # if sample['LIMIT_BAL'] > 135000.0
            # N. of samples 3248 ([2430.  818.])
            return 0
    else:  # if sample['PAY_AMT4'] > 0.5
        if sample['PAY_4'] <= -0.5:
            # N. of samples 6197 ([5382.  815.])
            return 0
        else:  # if sample['PAY_4'] > -0.5
            if sample['PAY_0'] <= 0.5:
                # N. of samples 13674 ([11960.  1714.])
                return 0
            else:  # if sample['PAY_0'] > 0.5
                # N. of samples 3721 ([1607. 2114.])
                return 1

@labeling_function()
def lf04(sample):
    if sample['PAY_0'] <= 1.5:
        if sample['PAY_0'] <= 0.5:
            if sample['PAY_AMT6'] <= 953.5:
                # N. of samples 8184 ([6676. 1508.])
                return 0
            else:  # if sample['PAY_AMT6'] > 953.5
                if sample['LIMIT_BAL'] <= 75000.0:
                    # N. of samples 3197 ([2638.  559.])
                    return 0
                else:  # if sample['LIMIT_BAL'] > 75000.0
                    # N. of samples 11801 ([10661.  1140.])
                    return 0
        else:  # if sample['PAY_0'] > 0.5
            # N. of samples 3688 ([2436. 1252.])
            return 0
    else:  # if sample['PAY_0'] > 1.5
        # N. of samples 3130 ([ 953. 2177.])
        return 1

@labeling_function()
def lf05(sample):
    if sample['PAY_6'] <= 1.0:
        if sample['PAY_AMT1'] <= 21.5:
            # N. of samples 4596 ([3171. 1425.])
            return 0
        else:  # if sample['PAY_AMT1'] > 21.5
            if sample['PAY_AMT2'] <= 1500.5:
                if sample['BILL_AMT3'] <= 1870.5:
                    # N. of samples 3311 ([2712.  599.])
                    return 0
                else:  # if sample['BILL_AMT3'] > 1870.5
                    # N. of samples 3666 ([2691.  975.])
                    return 0
            else:  # if sample['PAY_AMT2'] > 1500.5
                # N. of samples 15348 ([13322.  2026.])
                return 0
    else:  # if sample['PAY_6'] > 1.0
        # N. of samples 3079 ([1468. 1611.])
        return 1

@labeling_function()
def lf06(sample):
    if sample['PAY_AMT3'] <= 1.5:
        # N. of samples 5981 ([4046. 1935.])
        return 0
    else:  # if sample['PAY_AMT3'] > 1.5
        if sample['PAY_0'] <= 0.5:
            if sample['BILL_AMT5'] <= 31128.5:
                if sample['PAY_AMT4'] <= 2062.5:
                    # N. of samples 8217 ([6883. 1334.])
                    return 0
                else:  # if sample['PAY_AMT4'] > 2062.5
                    # N. of samples 3574 ([3251.  323.])
                    return 0
            else:  # if sample['BILL_AMT5'] > 31128.5
                # N. of samples 7822 ([7032.  790.])
                return 0
        else:  # if sample['PAY_0'] > 0.5
            # N. of samples 4406 ([2152. 2254.])
            return 1

@labeling_function()
def lf07(sample):
    if sample['PAY_6'] <= 1.0:
        if sample['BILL_AMT1'] <= 782.5:
            # N. of samples 4326 ([3280. 1046.])
            return 0
        else:  # if sample['BILL_AMT1'] > 782.5
            if sample['PAY_AMT1'] <= 656.0:
                # N. of samples 3016 ([2099.  917.])
                return 0
            else:  # if sample['PAY_AMT1'] > 656.0
                if sample['PAY_0'] <= -0.5:
                    # N. of samples 4379 ([3937.  442.])
                    return 0
                else:  # if sample['PAY_0'] > -0.5
                    # N. of samples 15200 ([12580.  2620.])
                    return 0
    else:  # if sample['PAY_6'] > 1.0
        # N. of samples 3079 ([1468. 1611.])
        return 1

@labeling_function()
def lf08(sample):
    if sample['PAY_AMT1'] <= 21.5:
        # N. of samples 5411 ([3469. 1942.])
        return 0
    else:  # if sample['PAY_AMT1'] > 21.5
        if sample['PAY_AMT3'] <= 2928.5:
            if sample['EDUCATION'] <= 1.5:
                # N. of samples 4368 ([3514.  854.])
                return 0
            else:  # if sample['EDUCATION'] > 1.5
                # N. of samples 10000 ([7637. 2363.])
                return 0
        else:  # if sample['PAY_AMT3'] > 2928.5
            if sample['PAY_AMT6'] <= 2501.0:
                # N. of samples 3148 ([2583.  565.])
                return 0
            else:  # if sample['PAY_AMT6'] > 2501.0
                # N. of samples 7073 ([6161.  912.])
                return 0

@labeling_function()
def lf09(sample):
    if sample['PAY_3'] <= 1.5:
        if sample['PAY_AMT6'] <= 1502.5:
            if sample['PAY_0'] <= -0.5:
                # N. of samples 4408 ([3604.  804.])
                return 0
            else:  # if sample['PAY_0'] > -0.5
                # N. of samples 8307 ([6428. 1879.])
                return 0
        else:  # if sample['PAY_AMT6'] > 1502.5
            if sample['LIMIT_BAL'] <= 145000.0:
                # N. of samples 4806 ([3912.  894.])
                return 0
            else:  # if sample['LIMIT_BAL'] > 145000.0
                # N. of samples 8270 ([7412.  858.])
                return 0
    else:  # if sample['PAY_3'] > 1.5
        # N. of samples 4209 ([2008. 2201.])
        return 1

@labeling_function()
def lf10(sample):
    if sample['PAY_0'] <= 1.5:
        if sample['LIMIT_BAL'] <= 75000.0:
            if sample['LIMIT_BAL'] <= 45000.0:
                # N. of samples 3458 ([2500.  958.])
                return 0
            else:  # if sample['LIMIT_BAL'] > 45000.0
                # N. of samples 4216 ([3376.  840.])
                return 0
        else:  # if sample['LIMIT_BAL'] > 75000.0
            if sample['PAY_AMT3'] <= 823.5:
                # N. of samples 5489 ([4300. 1189.])
                return 0
            else:  # if sample['PAY_AMT3'] > 823.5
                # N. of samples 13707 ([12235.  1472.])
                return 0
    else:  # if sample['PAY_0'] > 1.5
        # N. of samples 3130 ([ 953. 2177.])
        return 1
