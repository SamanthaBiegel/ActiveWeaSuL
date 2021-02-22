from snorkel.labeling import labeling_function

@labeling_function()
def lf01(sample):
    if sample['HumidityRatio'] <= 0.005165699403733015:
        if sample['CO2'] <= 617.25:
            # N. of samples 1205 ([1179.   26.])
            return 0
        else:  # if sample['CO2'] > 617.25
            if sample['CO2'] <= 778.9166564941406:
                # N. of samples 88 ([42. 46.])
                return 1
            else:  # if sample['CO2'] > 778.9166564941406
                # N. of samples 227 ([ 28. 199.])
                return 1
    else:  # if sample['HumidityRatio'] > 0.005165699403733015
        if sample['CO2'] <= 1307.5:
            # N. of samples 36 ([24. 12.])
            return 0
        else:  # if sample['CO2'] > 1307.5
            # N. of samples 73 ([10. 63.])
            return 1

@labeling_function()
def lf02(sample):
    if sample['date'] <= 1.423245990302843e+18:
        if sample['CO2'] <= 614.375:
            # N. of samples 325 ([308.  17.])
            return 0
        else:  # if sample['CO2'] > 614.375
            if sample['Light'] <= 187.58333206176758:
                # N. of samples 27 ([27.  0.])
                return 0
            else:  # if sample['Light'] > 187.58333206176758
                # N. of samples 238 ([ 16. 222.])
                return 1
    else:  # if sample['date'] > 1.423245990302843e+18
        if sample['Humidity'] <= 33.21125030517578:
            # N. of samples 908 ([874.  34.])
            return 0
        else:  # if sample['Humidity'] > 33.21125030517578
            # N. of samples 131 ([58. 73.])
            return 1

@labeling_function()
def lf03(sample):
    if sample['Temperature'] <= 21.26750087738037:
        # N. of samples 1184 ([1111.   73.])
        return 0
    else:  # if sample['Temperature'] > 21.26750087738037
        if sample['HumidityRatio'] <= 0.003045869176276028:
            # N. of samples 65 ([65.  0.])
            return 0
        else:  # if sample['HumidityRatio'] > 0.003045869176276028
            if sample['Humidity'] <= 21.833749771118164:
                # N. of samples 116 ([ 13. 103.])
                return 1
            else:  # if sample['Humidity'] > 21.833749771118164
                if sample['CO2'] <= 689.1666564941406:
                    # N. of samples 63 ([61.  2.])
                    return 0
                else:  # if sample['CO2'] > 689.1666564941406
                    # N. of samples 201 ([ 33. 168.])
                    return 1

@labeling_function()
def lf04(sample):
    if sample['date'] <= 1.423245990302843e+18:
        if sample['Temperature'] <= 21.62916660308838:
            # N. of samples 368 ([302.  66.])
            return 0
        else:  # if sample['Temperature'] > 21.62916660308838
            # N. of samples 222 ([ 49. 173.])
            return 1
    else:  # if sample['date'] > 1.423245990302843e+18
        if sample['Temperature'] <= 20.6875:
            # N. of samples 816 ([802.  14.])
            return 0
        else:  # if sample['Temperature'] > 20.6875
            if sample['Light'] <= 385.8333282470703:
                # N. of samples 131 ([130.   1.])
                return 0
            else:  # if sample['Light'] > 385.8333282470703
                # N. of samples 92 ([ 0. 92.])
                return 1

@labeling_function()
def lf05(sample):
    if sample['date'] <= 1.423245990302843e+18:
        if sample['date'] <= 1.423209019224359e+18:
            if sample['Temperature'] <= 21.954166412353516:
                # N. of samples 330 ([310.  20.])
                return 0
            else:  # if sample['Temperature'] > 21.954166412353516
                # N. of samples 122 ([32. 90.])
                return 1
        else:  # if sample['date'] > 1.423209019224359e+18
            # N. of samples 138 ([  9. 129.])
            return 1
    else:  # if sample['date'] > 1.423245990302843e+18
        if sample['Humidity'] <= 33.21125030517578:
            # N. of samples 908 ([874.  34.])
            return 0
        else:  # if sample['Humidity'] > 33.21125030517578
            # N. of samples 131 ([58. 73.])
            return 1

@labeling_function()
def lf06(sample):
    if sample['date'] <= 1.423245990302843e+18:
        if sample['Humidity'] <= 19.016666412353516:
            # N. of samples 56 ([54.  2.])
            return 0
        else:  # if sample['Humidity'] > 19.016666412353516
            if sample['Temperature'] <= 21.62916660308838:
                # N. of samples 313 ([248.  65.])
                return 0
            else:  # if sample['Temperature'] > 21.62916660308838
                # N. of samples 221 ([ 49. 172.])
                return 1
    else:  # if sample['date'] > 1.423245990302843e+18
        if sample['CO2'] <= 1100.75:
            # N. of samples 929 ([908.  21.])
            return 0
        else:  # if sample['CO2'] > 1100.75
            # N. of samples 110 ([24. 86.])
            return 1

@labeling_function()
def lf07(sample):
    if sample['HumidityRatio'] <= 0.005165699403733015:
        if sample['HumidityRatio'] <= 0.002999819116666913:
            # N. of samples 316 ([308.   8.])
            return 0
        else:  # if sample['HumidityRatio'] > 0.002999819116666913
            if sample['Humidity'] <= 21.833749771118164:
                # N. of samples 249 ([128. 121.])
                return 0
            else:  # if sample['Humidity'] > 21.833749771118164
                if sample['Light'] <= 307.0:
                    # N. of samples 799 ([799.   0.])
                    return 0
                else:  # if sample['Light'] > 307.0
                    # N. of samples 156 ([ 14. 142.])
                    return 1
    else:  # if sample['HumidityRatio'] > 0.005165699403733015
        # N. of samples 109 ([34. 75.])
        return 1

@labeling_function()
def lf08(sample):
    if sample['HumidityRatio'] <= 0.005165699403733015:
        if sample['Humidity'] <= 19.458749771118164:
            # N. of samples 311 ([298.  13.])
            return 0
        else:  # if sample['Humidity'] > 19.458749771118164
            if sample['Temperature'] <= 21.26750087738037:
                if sample['Temperature'] <= 20.524999618530273:
                    # N. of samples 694 ([682.  12.])
                    return 0
                else:  # if sample['Temperature'] > 20.524999618530273
                    # N. of samples 226 ([176.  50.])
                    return 0
            else:  # if sample['Temperature'] > 21.26750087738037
                # N. of samples 289 ([ 93. 196.])
                return 1
    else:  # if sample['HumidityRatio'] > 0.005165699403733015
        # N. of samples 109 ([34. 75.])
        return 1

@labeling_function()
def lf09(sample):
    if sample['Humidity'] <= 37.64500045776367:
        if sample['HumidityRatio'] <= 0.004358127247542143:
            if sample['Temperature'] <= 21.26750087738037:
                # N. of samples 973 ([932.  41.])
                return 0
            else:  # if sample['Temperature'] > 21.26750087738037
                # N. of samples 274 ([131. 143.])
                return 1
        else:  # if sample['HumidityRatio'] > 0.004358127247542143
            if sample['Temperature'] <= 20.68333339691162:
                # N. of samples 172 ([164.   8.])
                return 0
            else:  # if sample['Temperature'] > 20.68333339691162
                # N. of samples 174 ([ 56. 118.])
                return 1
    else:  # if sample['Humidity'] > 37.64500045776367
        # N. of samples 36 ([ 0. 36.])
        return 1

@labeling_function()
def lf10(sample):
    if sample['HumidityRatio'] <= 0.005165699403733015:
        if sample['HumidityRatio'] <= 0.002999819116666913:
            # N. of samples 316 ([308.   8.])
            return 0
        else:  # if sample['HumidityRatio'] > 0.002999819116666913
            if sample['date'] <= 1.423245990302843e+18:
                # N. of samples 505 ([274. 231.])
                return 0
            else:  # if sample['date'] > 1.423245990302843e+18
                if sample['Light'] <= 375.5:
                    # N. of samples 666 ([666.   0.])
                    return 0
                else:  # if sample['Light'] > 375.5
                    # N. of samples 33 ([ 1. 32.])
                    return 1
    else:  # if sample['HumidityRatio'] > 0.005165699403733015
        # N. of samples 109 ([34. 75.])
        return 1
