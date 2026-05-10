# Price Model Analysis: 6507aee

Ledger commit: `6507aee`
Setup commit: `6507aee`

## Metrics

| train_rmsle | val_rmsle | overfit_gap | val_spearman | val_mae | val_median_ae | val_p90_ae |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1730 | 0.2592 | 0.4977 | 0.7878 | 3.8750 | 0.9685 | 5.0986 |

## Validation Deciles

| decile | count | min_true | max_true | mean_true | mean_pred | mean_error | mae | median_ae | rmsle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 100 | 0.1670 | 4.7705 | 3.8242 | 5.0317 | 1.2074 | 1.4142 | 0.9982 | 0.3743 |
| 1 | 100 | 4.7802 | 5.4734 | 5.1582 | 5.7605 | 0.6023 | 1.0243 | 0.7067 | 0.2111 |
| 2 | 100 | 5.4794 | 5.9231 | 5.6934 | 6.1860 | 0.4925 | 0.9204 | 0.6391 | 0.1688 |
| 3 | 100 | 5.9367 | 6.4073 | 6.1537 | 6.4442 | 0.2906 | 0.8502 | 0.6205 | 0.1537 |
| 4 | 100 | 6.4082 | 6.8785 | 6.6438 | 6.7257 | 0.0819 | 0.7225 | 0.5446 | 0.1312 |
| 5 | 100 | 6.8964 | 7.4490 | 7.1680 | 7.0810 | -0.0870 | 0.8884 | 0.6982 | 0.1438 |
| 6 | 100 | 7.4515 | 8.4948 | 7.8964 | 7.5434 | -0.3530 | 1.3742 | 0.9473 | 0.2197 |
| 7 | 100 | 8.5443 | 10.8812 | 9.5176 | 9.3063 | -0.2113 | 1.8877 | 1.5668 | 0.2278 |
| 8 | 100 | 10.9267 | 18.6198 | 13.5880 | 12.7253 | -0.8627 | 3.2590 | 3.3006 | 0.3021 |
| 9 | 100 | 18.7135 | 2044.3314 | 56.2090 | 34.2552 | -21.9537 | 26.4093 | 7.8872 | 0.4534 |

## Worst Absolute Errors

| coffee_name | roaster | price_raw | true | pred | error | abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| Panama Kaizen Lot GW-01 | Paradise Roasters | $400.00/20 grams | 2044.3314 | 529.1223 | -1515.2091 | 1515.2091 |
| Panama Boquete El Trompo Geisha Natural (Sky Project 2000+) | Dou Zhai Coffee & Roastery | NT $1,000/20 grams | 155.2209 | 229.0708 | 73.8499 | 73.8499 |
| Civet Yirgacheffe Sisota | Good Chance Biotechnology, Ltd. | $206.00/8 ounces | 118.9973 | 57.2983 | -61.6989 | 61.6989 |
| Mama Cata Mokkita | Paradise Roasters | $150.00/4 ounces | 154.6202 | 99.1228 | -55.4974 | 55.4974 |
| Lotus by Ninety Plus | Dragonfly Coffee Roasters | $145.00/8 ounces | 83.3023 | 31.5685 | -51.7338 | 51.7338 |
| Taiwan Natural Alishan Zhuo-Wu Geisha | Kakalove Cafe | NT $600/50 grams | 41.0921 | 85.9667 | 44.8746 | 44.8746 |
| Gesha, Gesha Village 1931 | Mudhouse Coffee Roasters | $88.00/8 ounces | 51.8051 | 11.4171 | -40.3880 | 40.3880 |
| Kona Mokka Champagne Natural | Paradise Roasters | $100.00/4 ounces | 93.1234 | 63.5985 | -29.5249 | 29.5249 |
| Ninety Plus Gesha Estates Limited Batch #227 | JBC Coffee Roasters | $68.50/4 ounces | 80.6346 | 55.6841 | -24.9506 | 24.9506 |
| Panama Abu Natural ASD Geisha BOP GN11 | Kakalove Cafe | NT $1200/4 ounces | 46.2024 | 22.1085 | -24.0940 | 24.0940 |
| Coffea Diversa Cioiccie | Paradise Roasters | $39.95/4 ounces | 44.8610 | 21.2244 | -23.6366 | 23.6366 |
| Single-Origin Ethiopia Espresso | Cafe330 | HKD $428/150 grams | 39.9340 | 16.4595 | -23.4745 | 23.4745 |
| Gesha Village Natural Oma Block Lot 72 | Kakalove Cafe | NT $3200/16 ounces | 30.9870 | 9.3458 | -21.6411 | 21.6411 |
| Colombia Valle del Cauca Cerro Azul Geisha AAA | Victrola Coffee Roasters | $59.75/8 ounces | 36.7922 | 15.9346 | -20.8576 | 20.8576 |
| Panama Finca Deborah Afterglow Geisha Natural | Mute Roaster | HKD $498/105 grams | 67.5437 | 48.2180 | -19.3257 | 19.3257 |

## Largest Log Errors

| coffee_name | roaster | price_raw | true | pred | log_error |
| --- | --- | --- | --- | --- | --- |
| Koffee Kool 002 (KK002) | Yum China | 110 CN¥/500 grams | 0.1670 | 4.5111 | 1.5523 |
| Gesha, Gesha Village 1931 | Mudhouse Coffee Roasters | $88.00/8 ounces | 51.8051 | 11.4171 | -1.4475 |
| Honey-Processed Yunnan Espresso | Chu Bei | ¥88/16 ounces | 0.1987 | 3.6559 | 1.3569 |
| Panama Kaizen Lot GW-01 | Paradise Roasters | $400.00/20 grams | 2044.3314 | 529.1223 | -1.3502 |
| Gesha Village Natural Oma Block Lot 72 | Kakalove Cafe | NT $3200/16 ounces | 30.9870 | 9.3458 | -1.1287 |
| Elida Natural ASD (Panama) | Geisha Coffee Roaster | $14.74/16 ounces | 3.9476 | 13.2022 | 1.0545 |
| Honduras Catuai/Bourbon | Spirit Animal Coffee | $39.00/12 ounces | 14.2285 | 4.7586 | -0.9725 |
| Hawaii Ka’u Rusty’s Typica Washed | Bacca Cafe | NT $2500/16 ounces | 26.3497 | 9.4063 | -0.9663 |
| Lotus by Ninety Plus | Dragonfly Coffee Roasters | $145.00/8 ounces | 83.3023 | 31.5685 | -0.9511 |
| Finca Santa Elena Bourbon Honey | Caoban Coffee | NT $170/100 grams | 5.3951 | 15.1764 | 0.9280 |
| San Formosan Estate | Nantou Specialty | NTD $1800/16 ounces | 18.5813 | 6.8115 | -0.9190 |
| Finca El Diamante Borbón COE | El Gran Cafe | $47.00/12 ounces | 13.8917 | 5.0566 | -0.8997 |
| Single-Origin Ethiopia Espresso | Cafe330 | HKD $428/150 grams | 39.9340 | 16.4595 | -0.8521 |
| 100% Arabica 100% Italiano | Caffe Bomrad | $54.00/1 Kilogram | 7.7500 | 2.8555 | -0.8196 |
| Colombia Valle del Cauca Cerro Azul Geisha AAA | Victrola Coffee Roasters | $59.75/8 ounces | 36.7922 | 15.9346 | -0.8027 |

## High-RMSLE Slices

### Origin Country
| value | count | mean_true | mean_pred | mean_error | mae | rmsle |
| --- | --- | --- | --- | --- | --- | --- |
| Panama | 48 | 74.6818 | 41.7076 | -32.9742 | 41.4972 | 0.4276 |
| unknown | 87 | 14.9292 | 13.4732 | -1.4560 | 3.5142 | 0.2978 |
| blend_multi_origin | 87 | 5.9038 | 5.9603 | 0.0565 | 1.2460 | 0.2784 |
| Brazil | 23 | 8.4087 | 7.8474 | -0.5614 | 2.1113 | 0.2686 |
| Ethiopia | 235 | 8.6042 | 7.8845 | -0.7197 | 2.2003 | 0.2607 |
| Colombia | 102 | 10.5348 | 9.7174 | -0.8174 | 2.4106 | 0.2527 |
| Guatemala | 57 | 8.3727 | 7.6309 | -0.7419 | 1.4312 | 0.2288 |
| El Salvador | 33 | 7.3028 | 7.3020 | -0.0008 | 1.4016 | 0.2279 |
| Costa Rica | 47 | 8.4822 | 7.7786 | -0.7035 | 1.4929 | 0.1852 |
| Indonesia | 42 | 6.3605 | 6.1252 | -0.2353 | 0.9139 | 0.1746 |
| Kenya | 88 | 8.3530 | 8.0652 | -0.2878 | 1.3265 | 0.1692 |

### Roaster Country
| value | count | mean_true | mean_pred | mean_error | mae | rmsle |
| --- | --- | --- | --- | --- | --- | --- |
| Taiwan | 227 | 11.3087 | 10.9847 | -0.3240 | 3.6045 | 0.3234 |
| Canada | 30 | 5.6876 | 5.3611 | -0.3265 | 1.1206 | 0.2298 |
| United States | 688 | 12.7853 | 10.0083 | -2.7770 | 4.1027 | 0.2105 |

## Feature Importance

Nonzero coefficients: 9310 / 21498

### Top Positive Coefficients
| feature | coef |
| --- | --- |
| tfidf:kona | 1.4444 |
| tfidf:civet | 1.1937 |
| tfidf:luwak | 1.0387 |
| tfidf:rare | 1.0111 |
| variety=mokkita | 0.9316 |
| tfidf:kopi | 0.7790 |
| tfidf:kopi luwak | 0.7790 |
| tfidf:civet cat | 0.7576 |
| tfidf:cat | 0.7341 |
| origin_country=Jamaica | 0.6385 |
| tfidf:kona coffee | 0.6239 |
| tfidf:located the | 0.5917 |
| tfidf:competition | 0.5891 |
| tfidf:sidra | 0.5781 |
| tfidf:ltd is | 0.5751 |
| tfidf:finish carries | 0.5652 |
| tfidf:paradise prides | 0.5649 |
| tfidf:first | 0.5519 |
| tfidf:subtle | 0.5513 |
| origin_country=Yemen | 0.5446 |
| tfidf:808-498-4048 | 0.5394 |
| tfidf:808-498-4048 for | 0.5394 |
| tfidf:call 808-498-4048 | 0.5394 |
| tfidf:island | 0.5317 |
| tfidf:lab | 0.5080 |

### Top Negative Coefficients
| feature | coef |
| --- | --- |
| tfidf:roasting of | -0.7500 |
| tfidf:tea visit | -0.6989 |
| tfidf:and tea | -0.6983 |
| tfidf:value | -0.6907 |
| tfidf:ba | -0.6828 |
| tfidf:and cafe | -0.6585 |
| tfidf:robusta | -0.6567 |
| tfidf:store | -0.6212 |
| tfidf:euphora coffee | -0.5942 |
| tfidf:nutella and | -0.5926 |
| tfidf:ready-to-drink | -0.5870 |
| tfidf:tested | -0.5808 |
| tfidf:yang | -0.5478 |
| tfidf:cacao and | -0.5438 |
| tfidf:and roast | -0.5422 |
| tfidf:coffee tested | -0.5377 |
| roaster_country=Peru | -0.5257 |
| tfidf:taiwan visit | -0.5160 |
| tfidf:roastery is | -0.5077 |
| tfidf:at beneficio | -0.5020 |
| tfidf:products | -0.4946 |
| tfidf:roasted | -0.4866 |
| tfidf:earthy | -0.4859 |
| tfidf:retail locations | -0.4851 |
| tfidf:roaster and | -0.4788 |
