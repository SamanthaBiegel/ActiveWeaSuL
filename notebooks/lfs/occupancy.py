from snorkel.labeling import labeling_function

@labeling_function()
def lf01(sample):
    if sample['CO2'] <= 629.5416564941406:
        # N. of samples 4856 ([2898. 1958.])
        return 0
    else:  # if sample['CO2'] > 629.5416564941406
        if sample['Humidity'] <= 19.08750057220459:
            # N. of samples 4 ([0. 4.])
            return 1
        else:  # if sample['Humidity'] > 19.08750057220459
            if sample['CO2'] <= 800.125:
                # N. of samples 411 ([212. 199.])
                return 0
            else:  # if sample['CO2'] > 800.125
                if sample['HumidityRatio'] <= 0.003911018371582031:
                    # N. of samples 266 ([101. 165.])
                    return 1
                else:  # if sample['HumidityRatio'] > 0.003911018371582031
                    # N. of samples 977 ([448. 529.])
                    return 1

@labeling_function()
def lf02(sample):
    if sample['date'] <= 1.4232453031080755e+18:
        if sample['date'] <= 1.4232119054423818e+18:
            if sample['Temperature'] <= 21.873332977294922:
                # N. of samples 1332 ([766. 566.])
                return 0
            else:  # if sample['Temperature'] > 21.873332977294922
                # N. of samples 535 ([254. 281.])
                return 1
        else:  # if sample['date'] > 1.4232119054423818e+18
            # N. of samples 430 ([181. 249.])
            return 1
    else:  # if sample['date'] > 1.4232453031080755e+18
        if sample['Humidity'] <= 33.25625038146973:
            # N. of samples 3705 ([2215. 1490.])
            return 0
        else:  # if sample['Humidity'] > 33.25625038146973
            # N. of samples 512 ([243. 269.])
            return 1

@labeling_function()
def lf03(sample):
    if sample['Temperature'] <= 21.21125030517578:
        # N. of samples 4670 ([2756. 1914.])
        return 0
    else:  # if sample['Temperature'] > 21.21125030517578
        if sample['HumidityRatio'] <= 0.003057029447518289:
            if sample['Humidity'] <= 19.130000114440918:
                # N. of samples 291 ([170. 121.])
                return 0
            else:  # if sample['Humidity'] > 19.130000114440918
                # N. of samples 8 ([8. 0.])
                return 0
        else:  # if sample['HumidityRatio'] > 0.003057029447518289
            if sample['CO2'] <= 800.125:
                # N. of samples 505 ([283. 222.])
                return 0
            else:  # if sample['CO2'] > 800.125
                # N. of samples 1040 ([442. 598.])
                return 1

@labeling_function()
def lf04(sample):
    if sample['date'] <= 1.4232453031080755e+18:
        if sample['Temperature'] <= 21.72624969482422:
            # N. of samples 1545 ([859. 686.])
            return 0
        else:  # if sample['Temperature'] > 21.72624969482422
            # N. of samples 752 ([342. 410.])
            return 1
    else:  # if sample['date'] > 1.4232453031080755e+18
        if sample['Temperature'] <= 21.2810001373291:
            # N. of samples 3561 ([2126. 1435.])
            return 0
        else:  # if sample['Temperature'] > 21.2810001373291
            if sample['Light'] <= 427.1666717529297:
                # N. of samples 365 ([215. 150.])
                return 0
            else:  # if sample['Light'] > 427.1666717529297
                # N. of samples 291 ([117. 174.])
                return 1

@labeling_function()
def lf05(sample):
    if sample['HumidityRatio'] <= 0.005187563365325332:
        if sample['HumidityRatio'] <= 0.002908413182012737:
            # N. of samples 1023 ([624. 399.])
            return 0
        else:  # if sample['HumidityRatio'] > 0.002908413182012737
            if sample['Humidity'] <= 24.922082901000977:
                # N. of samples 1721 ([903. 818.])
                return 0
            else:  # if sample['Humidity'] > 24.922082901000977
                if sample['Light'] <= 207.25:
                    # N. of samples 2741 ([1674. 1067.])
                    return 0
                else:  # if sample['Light'] > 207.25
                    # N. of samples 604 ([267. 337.])
                    return 1
    else:  # if sample['HumidityRatio'] > 0.005187563365325332
        # N. of samples 425 ([191. 234.])
        return 1

@labeling_function()
def lf06(sample):
    if sample['HumidityRatio'] <= 0.005187563365325332:
        if sample['Humidity'] <= 19.935832977294922:
            # N. of samples 1556 ([927. 629.])
            return 0
        else:  # if sample['Humidity'] > 19.935832977294922
            if sample['Temperature'] <= 21.749500274658203:
                if sample['Temperature'] <= 20.54166603088379:
                    # N. of samples 2598 ([1557. 1041.])
                    return 0
                else:  # if sample['Temperature'] > 20.54166603088379
                    # N. of samples 1234 ([666. 568.])
                    return 0
            else:  # if sample['Temperature'] > 21.749500274658203
                # N. of samples 701 ([318. 383.])
                return 1
    else:  # if sample['HumidityRatio'] > 0.005187563365325332
        # N. of samples 425 ([191. 234.])
        return 1

@labeling_function()
def lf07(sample):
    if sample['HumidityRatio'] <= 0.005187563365325332:
        if sample['HumidityRatio'] <= 0.002908413182012737:
            # N. of samples 1023 ([624. 399.])
            return 0
        else:  # if sample['HumidityRatio'] > 0.002908413182012737
            if sample['date'] <= 1.4232453031080755e+18:
                # N. of samples 2025 ([1042.  983.])
                return 0
            else:  # if sample['date'] > 1.4232453031080755e+18
                if sample['Light'] <= 377.375:
                    # N. of samples 2856 ([1721. 1135.])
                    return 0
                else:  # if sample['Light'] > 377.375
                    # N. of samples 185 ([ 81. 104.])
                    return 1
    else:  # if sample['HumidityRatio'] > 0.005187563365325332
        # N. of samples 425 ([191. 234.])
        return 1

@labeling_function()
def lf08(sample):
    if sample['date'] <= 1.4232453031080755e+18:
        # N. of samples 2297 ([1201. 1096.])
        return 0
    else:  # if sample['date'] > 1.4232453031080755e+18
        if sample['CO2'] <= 1200.125:
            if sample['Humidity'] <= 28.04166603088379:
                if sample['Humidity'] <= 27.1875:
                    # N. of samples 1925 ([1162.  763.])
                    return 0
                else:  # if sample['Humidity'] > 27.1875
                    # N. of samples 397 ([269. 128.])
                    return 0
            else:  # if sample['Humidity'] > 28.04166603088379
                # N. of samples 1502 ([857. 645.])
                return 0
        else:  # if sample['CO2'] > 1200.125
            # N. of samples 393 ([170. 223.])
            return 1

@labeling_function()
def lf09(sample):
    if sample['Humidity'] <= 33.25625038146973:
        if sample['HumidityRatio'] <= 0.004213790874928236:
            if sample['Temperature'] <= 20.58750057220459:
                # N. of samples 2402 ([1480.  922.])
                return 0
            else:  # if sample['Temperature'] > 20.58750057220459
                # N. of samples 1842 ([987. 855.])
                return 0
        else:  # if sample['HumidityRatio'] > 0.004213790874928236
            # N. of samples 1758 ([949. 809.])
            return 0
    else:  # if sample['Humidity'] > 33.25625038146973
        if sample['Temperature'] <= 21.315000534057617:
            # N. of samples 211 ([118.  93.])
            return 0
        else:  # if sample['Temperature'] > 21.315000534057617
            # N. of samples 301 ([125. 176.])
            return 1

@labeling_function()
def lf10(sample):
    if sample['Humidity'] <= 33.25625038146973:
        if sample['Humidity'] <= 19.935832977294922:
            # N. of samples 1556 ([927. 629.])
            return 0
        else:  # if sample['Humidity'] > 19.935832977294922
            # N. of samples 4446 ([2489. 1957.])
            return 0
    else:  # if sample['Humidity'] > 33.25625038146973
        if sample['CO2'] <= 1223.6666870117188:
            # N. of samples 195 ([106.  89.])
            return 0
        else:  # if sample['CO2'] > 1223.6666870117188
            if sample['HumidityRatio'] <= 0.005411889869719744:
                # N. of samples 63 ([16. 47.])
                return 1
            else:  # if sample['HumidityRatio'] > 0.005411889869719744
                # N. of samples 254 ([121. 133.])
                return 1
