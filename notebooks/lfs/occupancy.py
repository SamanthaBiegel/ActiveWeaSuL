from snorkel.labeling import labeling_function

@labeling_function()
def lf01(sample):
    if sample['CO2'] <= 614.625:
        if sample['Light'] <= 376.75:
            if sample['Humidity'] <= 22.364999771118164:
                # N. of samples 2173 ([2173.    0.])
                # N. of samples 3694 ([3692.    2.])
                return 0
        else:  # if sample['Light'] > 376.75
            if sample['CO2'] <= 456.3333282470703:
                # N. of samples 11 ([10.  1.])
                return 0
            else:  # if sample['CO2'] > 456.3333282470703
                # N. of samples 142 ([  9. 133.])
                return 1
    else:  # if sample['CO2'] > 614.625
        if sample['date'] <= 1.4235050627301376e+18:
            if sample['Light'] <= 188.25:
                # N. of samples 210 ([209.   1.])
                return 0
            else:  # if sample['Light'] > 188.25
                # N. of samples 1639 ([  71. 1568.])
                return 1
        else:  # if sample['date'] > 1.4235050627301376e+18
            if sample['Temperature'] <= 20.7787504196167:
                # N. of samples 166 ([166.   0.])
                return 0
            else:  # if sample['Temperature'] > 20.7787504196167
                # N. of samples 108 ([84. 24.])
                return -1
    return -1

            
@labeling_function()
def lf02(sample):
    if sample['CO2'] <= 614.625:
        if sample['Humidity'] <= 33.44875144958496:
            if sample['Light'] <= 376.75:
                # N. of samples 5867 ([5.865e+03 2.000e+00])
                return 0
            else:  # if sample['Light'] > 376.75
                # N. of samples 125 ([ 19. 106.])
                return 1
        else:  # if sample['Humidity'] > 33.44875144958496
            # N. of samples 28 ([ 0. 28.])
            return 1
    else:  # if sample['CO2'] > 614.625
        if sample['Temperature'] <= 20.7787504196167:
            if sample['Light'] <= 198.5:
                # N. of samples 166 ([166.   0.])
                return 0
            else:  # if sample['Light'] > 198.5
                # N. of samples 66 ([ 0. 66.])
                return 1
        else:  # if sample['Temperature'] > 20.7787504196167
            if sample['CO2'] <= 755.1666564941406:
                # N. of samples 316 ([145. 171.])
                return -1
            else:  # if sample['CO2'] > 755.1666564941406
                # N. of samples 1575 ([ 219. 1356.])
                return 1

@labeling_function()
def lf03(sample):
    if sample['CO2'] <= 614.625:
        if sample['Humidity'] <= 33.44875144958496:
            if sample['Light'] <= 376.75:
                # N. of samples 5867 ([5.865e+03 2.000e+00])
                return 0
            else:  # if sample['Light'] > 376.75
                # N. of samples 125 ([ 19. 106.])
                return 1
        else:  # if sample['Humidity'] > 33.44875144958496
            # N. of samples 28 ([ 0. 28.])
            return 1
    else:  # if sample['CO2'] > 614.625
        if sample['date'] <= 1.4235050627301376e+18:
            if sample['Light'] <= 188.25:
                # N. of samples 210 ([209.   1.])
                return 0
            else:  # if sample['Light'] > 188.25
                # N. of samples 1639 ([  71. 1568.])
                return 1
        else:  # if sample['date'] > 1.4235050627301376e+18
            if sample['Light'] <= 213.0:
                # N. of samples 250 ([250.   0.])
                return 0
            else:  # if sample['Light'] > 213.0
                # N. of samples 24 ([ 0. 24.])
                return 1

@labeling_function()
def lf04(sample):
    if sample['CO2'] <= 614.625:
        if sample['HumidityRatio'] <= 0.004973354283720255:
            if sample['CO2'] <= 496.125:
                # N. of samples 5532 ([5507.   25.])
                return 0
            else:  # if sample['CO2'] > 496.125
                # N. of samples 464 ([377.  87.])
                return 0
        else:  # if sample['HumidityRatio'] > 0.004973354283720255
            # N. of samples 24 ([ 0. 24.])
            return 1
    else:  # if sample['CO2'] > 614.625
        if sample['date'] <= 1.4235050627301376e+18:
            if sample['date'] <= 1.4231000988537324e+18:
                # N. of samples 65 ([49. 16.])
                return -1
            else:  # if sample['date'] > 1.4231000988537324e+18
                # N. of samples 1784 ([ 231. 1553.])
                return 1
        else:  # if sample['date'] > 1.4235050627301376e+18
            if sample['date'] <= 1.4235397660658893e+18:
                # N. of samples 250 ([250.   0.])
                return 0
            else:  # if sample['date'] > 1.4235397660658893e+18
                # N. of samples 24 ([ 0. 24.])
                return 1

@labeling_function()
def lf05(sample):
    if sample['Temperature'] <= 21.252500534057617:
        if sample['Light'] <= 336.625:
            # N. of samples 1641 ([1.64e+03 1.00e+00])
            # N. of samples 3830 ([3830.    0.])
            return 0
        else:  # if sample['Light'] > 336.625
            if sample['Light'] <= 557.75:
                # N. of samples 395 ([ 15. 380.])
                return 1
            else:  # if sample['Light'] > 557.75
                # N. of samples 4 ([4. 0.])
                return 0
    else:  # if sample['Temperature'] > 21.252500534057617
        if sample['Humidity'] <= 18.96916675567627:
            # N. of samples 348 ([348.   0.])
            return 0
        else:  # if sample['Humidity'] > 18.96916675567627
            if sample['Humidity'] <= 21.84500026702881:
                # N. of samples 534 ([ 71. 463.])
                return 1
            else:  # if sample['Humidity'] > 21.84500026702881
                # N. of samples 1391 ([506. 885.])
                return -1

@labeling_function()
def lf06(sample):
    if sample['CO2'] <= 614.625:
        if sample['date'] <= 1.4235574269714104e+18:
            if sample['CO2'] <= 496.125:
                # N. of samples 5524 ([5505.   19.])
                return 0
            else:  # if sample['CO2'] > 496.125
                # N. of samples 464 ([377.  87.])
                return 0
        else:  # if sample['date'] > 1.4235574269714104e+18
            if sample['CO2'] <= 482.4166717529297:
                # N. of samples 4 ([2. 2.])
                return -1
            else:  # if sample['CO2'] > 482.4166717529297
                # N. of samples 28 ([ 0. 28.])
                return 1
    else:  # if sample['CO2'] > 614.625
        if sample['date'] <= 1.4235050627301376e+18:
            if sample['CO2'] <= 776.625:
                # N. of samples 358 ([157. 201.])
                return -1
            else:  # if sample['CO2'] > 776.625
                # N. of samples 1491 ([ 123. 1368.])
                return 1
        else:  # if sample['date'] > 1.4235050627301376e+18
            if sample['Temperature'] <= 20.7787504196167:
                # N. of samples 166 ([166.   0.])
                return 0
            else:  # if sample['Temperature'] > 20.7787504196167
                # N. of samples 108 ([84. 24.])
                return -1

@labeling_function()
def lf07(sample):
    if sample['Light'] <= 365.125:
        if sample['Light'] <= 209.25:
            # N. of samples 5540 ([5540.    0.])
            # N. of samples 523 ([522.   1.])
            return 0
        else:  # if sample['Light'] > 209.25
            if sample['HumidityRatio'] <= 0.004340938990935683:
                # N. of samples 265 ([261.   4.])
                return 0
            else:  # if sample['HumidityRatio'] > 0.004340938990935683
                # N. of samples 5 ([1. 4.])
                return 1
    else:  # if sample['Light'] > 365.125
        if sample['CO2'] <= 456.3333282470703:
            if sample['CO2'] <= 439.875:
                # N. of samples 2 ([1. 1.])
                return 0
            else:  # if sample['CO2'] > 439.875
                # N. of samples 9 ([9. 0.])
                return 0
        else:  # if sample['CO2'] > 456.3333282470703
            # N. of samples 384 ([ 64. 320.])
            # N. of samples 1415 ([  16. 1399.])
            return 1

@labeling_function()
def lf08(sample):
    if sample['HumidityRatio'] <= 0.005089321872219443:
        if sample['CO2'] <= 626.5833435058594:
            if sample['CO2'] <= 494.0416717529297:
                # N. of samples 5528 ([5502.   26.])
                return 0
            else:  # if sample['CO2'] > 494.0416717529297
                # N. of samples 522 ([407. 115.])
                return -1
        else:  # if sample['CO2'] > 626.5833435058594
            if sample['Humidity'] <= 33.08125114440918:
                # N. of samples 1389 ([ 239. 1150.])
                return 1
            else:  # if sample['Humidity'] > 33.08125114440918
                # N. of samples 78 ([78.  0.])
                return 0
    else:  # if sample['HumidityRatio'] > 0.005089321872219443
        if sample['Light'] <= 188.25:
            # N. of samples 1 ([0. 1.])
            # N. of samples 188 ([188.   0.])
            return 0
        else:  # if sample['Light'] > 188.25
            # N. of samples 437 ([  0. 437.])
            return 1

@labeling_function()
def lf09(sample):
    if sample['Light'] <= 365.125:
        if sample['HumidityRatio'] <= 0.0062260557897388935:
            # N. of samples 5478 ([5476. 2.])
            # N. of samples 853 ([848.   5.])
            return 0
        else:  # if sample['HumidityRatio'] > 0.0062260557897388935
            # N. of samples 2 ([0. 2.])
            return 1
    else:  # if sample['Light'] > 365.125
        if sample['date'] <= 1.4231455224278548e+18:
            # N. of samples 129 ([  5. 124.])
            # N. of samples 255 ([ 59. 196.])
            return -1
        else:  # if sample['date'] > 1.4231455224278548e+18
            if sample['CO2'] <= 456.3333282470703:
                # N. of samples 11 ([10.  1.])
                return 0
            else:  # if sample['CO2'] > 456.3333282470703
                # N. of samples 1415 ([  16. 1399.])
                return 1

@labeling_function()
def lf10(sample):
    if sample['Temperature'] <= 21.252500534057617:
        if sample['Light'] <= 336.625:
            # N. of samples 3171 ([3171.    0.])
            # N. of samples 2300 ([2299.  1.])
            return 0
        else:  # if sample['Light'] > 336.625
            if sample['CO2'] <= 456.3333282470703:
                # N. of samples 11 ([10.  1.])
                return -1
            else:  # if sample['CO2'] > 456.3333282470703
                # N. of samples 388 ([  9. 379.])
                return 1
    else:  # if sample['Temperature'] > 21.252500534057617
        if sample['HumidityRatio'] <= 0.003046866739168763:
            if sample['Humidity'] <= 19.29208278656006:
                # N. of samples 355 ([354.   1.])
                return 0
            else:  # if sample['Humidity'] > 19.29208278656006
                # N. of samples 1 ([0. 1.])
                return 1
        else:  # if sample['HumidityRatio'] > 0.003046866739168763
            if sample['Light'] <= 170.41666412353516:
                # N. of samples 500 ([499.   1.])
                return 0
            else:  # if sample['Light'] > 170.41666412353516
                # N. of samples 1417 ([  72. 1345.])
                return 1
            
labeling_functions = [lf01, lf02, lf03, lf04, lf05, lf06, lf07, lf08, lf09, lf10]