var PrimoDecisionTreeClassifier = function () {

    var findMax = function (nums) {
        var index = 0;
        for (var i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    this.predict = function (features) {
        var classes = new Array(6);

        if (features[0] <= 0.22674418985843658) {
            if (features[0] <= 0.12209302186965942) {
                if (features[1] <= 1.3932170271873474) {
                    if (features[9] <= 0.1141522042453289) {
                        if (features[1] <= 1.2912920117378235) {
                            if (features[16] <= 0.13425397872924805) {
                                if (features[1] <= 1.1715336441993713) {
                                    classes[0] = 256;
                                    classes[1] = 14;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 109;
                                    classes[1] = 106;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            } else {
                                classes[0] = 16783;
                                classes[1] = 243;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        } else {
                            if (features[9] <= 0.08413419499993324) {
                                classes[0] = 1666;
                                classes[1] = 200;
                                classes[2] = 1;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                classes[0] = 345;
                                classes[1] = 318;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        }
                    } else {
                        if (features[1] <= 1.0073650479316711) {
                            if (features[9] <= 0.1857825592160225) {
                                classes[0] = 1881;
                                classes[1] = 163;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                classes[0] = 70;
                                classes[1] = 300;
                                classes[2] = 0;
                                classes[3] = 1;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        } else {
                            if (features[1] <= 1.2064783573150635) {
                                if (features[9] <= 0.14443384110927582) {
                                    classes[0] = 303;
                                    classes[1] = 148;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 60;
                                    classes[1] = 366;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            } else {
                                classes[0] = 119;
                                classes[1] = 992;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        }
                    }
                } else {
                    if (features[9] <= 0.07323788478970528) {
                        if (features[1] <= 1.5516691207885742) {
                            classes[0] = 1126;
                            classes[1] = 619;
                            classes[2] = 0;
                            classes[3] = 0;
                            classes[4] = 0;
                            classes[5] = 0;
                        } else {
                            classes[0] = 108;
                            classes[1] = 638;
                            classes[2] = 0;
                            classes[3] = 0;
                            classes[4] = 0;
                            classes[5] = 0;
                        }
                    } else {
                        if (features[1] <= 1.4702287912368774) {
                            if (features[9] <= 0.0948193147778511) {
                                classes[0] = 242;
                                classes[1] = 222;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                classes[0] = 78;
                                classes[1] = 576;
                                classes[2] = 0;
                                classes[3] = 1;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        } else {
                            classes[0] = 115;
                            classes[1] = 2085;
                            classes[2] = 0;
                            classes[3] = 0;
                            classes[4] = 0;
                            classes[5] = 0;
                        }
                    }
                }
            } else {
                if (features[1] <= 1.047237515449524) {
                    if (features[17] <= 0.2022603079676628) {
                        if (features[17] <= 0.09030840918421745) {
                            classes[0] = 0;
                            classes[1] = 58;
                            classes[2] = 3;
                            classes[3] = 111;
                            classes[4] = 0;
                            classes[5] = 0;
                        } else {
                            classes[0] = 112;
                            classes[1] = 2142;
                            classes[2] = 1;
                            classes[3] = 40;
                            classes[4] = 0;
                            classes[5] = 0;
                        }
                    } else {
                        if (features[8] <= 0.1006128266453743) {
                            if (features[1] <= 0.9681802690029144) {
                                classes[0] = 4603;
                                classes[1] = 219;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                if (features[9] <= 0.0748910941183567) {
                                    classes[0] = 596;
                                    classes[1] = 180;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 117;
                                    classes[1] = 241;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            }
                        } else {
                            if (features[9] <= 0.08766145631670952) {
                                classes[0] = 276;
                                classes[1] = 138;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                classes[0] = 155;
                                classes[1] = 777;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        }
                    }
                } else {
                    if (features[9] <= 0.1885613352060318) {
                        if (features[1] <= 2.3657642602920532) {
                            if (features[17] <= 0.7467799782752991) {
                                if (features[1] <= 2.0532160997390747) {
                                    if (features[1] <= 1.1455852389335632) {
                                        if (features[9] <= 0.07125413417816162) {
                                            classes[0] = 298;
                                            classes[1] = 471;
                                            classes[2] = 0;
                                            classes[3] = 0;
                                            classes[4] = 0;
                                            classes[5] = 0;
                                        } else {
                                            classes[0] = 149;
                                            classes[1] = 1716;
                                            classes[2] = 0;
                                            classes[3] = 0;
                                            classes[4] = 0;
                                            classes[5] = 0;
                                        }
                                    } else {
                                        classes[0] = 300;
                                        classes[1] = 46651;
                                        classes[2] = 20;
                                        classes[3] = 530;
                                        classes[4] = 0;
                                        classes[5] = 0;
                                    }
                                } else {
                                    if (features[9] <= 0.14779949933290482) {
                                        classes[0] = 0;
                                        classes[1] = 5040;
                                        classes[2] = 10;
                                        classes[3] = 419;
                                        classes[4] = 0;
                                        classes[5] = 0;
                                    } else {
                                        classes[0] = 0;
                                        classes[1] = 1341;
                                        classes[2] = 13;
                                        classes[3] = 585;
                                        classes[4] = 0;
                                        classes[5] = 0;
                                    }
                                }
                            } else {
                                if (features[1] <= 1.2813449501991272) {
                                    classes[0] = 420;
                                    classes[1] = 183;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 17;
                                    classes[1] = 328;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            }
                        } else {
                            if (features[1] <= 2.5677891969680786) {
                                if (features[9] <= 0.13914143294095993) {
                                    classes[0] = 0;
                                    classes[1] = 1017;
                                    classes[2] = 5;
                                    classes[3] = 342;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 343;
                                    classes[2] = 4;
                                    classes[3] = 543;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            } else {
                                classes[0] = 0;
                                classes[1] = 250;
                                classes[2] = 4;
                                classes[3] = 703;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        }
                    } else {
                        if (features[1] <= 1.718815565109253) {
                            if (features[9] <= 0.2758122980594635) {
                                classes[0] = 0;
                                classes[1] = 2803;
                                classes[2] = 10;
                                classes[3] = 287;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                if (features[9] <= 0.3209354430437088) {
                                    if (features[1] <= 1.3587872385978699) {
                                        classes[0] = 0;
                                        classes[1] = 150;
                                        classes[2] = 0;
                                        classes[3] = 36;
                                        classes[4] = 0;
                                        classes[5] = 0;
                                    } else {
                                        classes[0] = 0;
                                        classes[1] = 78;
                                        classes[2] = 2;
                                        classes[3] = 167;
                                        classes[4] = 0;
                                        classes[5] = 0;
                                    }
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 25;
                                    classes[2] = 2;
                                    classes[3] = 183;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            }
                        } else {
                            if (features[1] <= 1.9871243238449097) {
                                if (features[9] <= 0.234668530523777) {
                                    classes[0] = 0;
                                    classes[1] = 819;
                                    classes[2] = 2;
                                    classes[3] = 415;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 125;
                                    classes[2] = 7;
                                    classes[3] = 493;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            } else {
                                if (features[9] <= 0.21278491616249084) {
                                    classes[0] = 0;
                                    classes[1] = 268;
                                    classes[2] = 7;
                                    classes[3] = 560;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 84;
                                    classes[2] = 3;
                                    classes[3] = 841;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (features[0] <= 0.8313953578472137) {
                if (features[9] <= 0.1186702698469162) {
                    if (features[1] <= 1.447826087474823) {
                        if (features[17] <= 1.2809196710586548) {
                            if (features[1] <= 0.700141578912735) {
                                if (features[17] <= 1.0918862223625183) {
                                    classes[0] = 22;
                                    classes[1] = 122;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 86;
                                    classes[1] = 18;
                                    classes[2] = 0;
                                    classes[3] = 0;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            } else {
                                classes[0] = 46;
                                classes[1] = 1518;
                                classes[2] = 12;
                                classes[3] = 232;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        } else {
                            if (features[1] <= 0.8989198505878448) {
                                classes[0] = 386;
                                classes[1] = 11;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            } else {
                                classes[0] = 66;
                                classes[1] = 91;
                                classes[2] = 0;
                                classes[3] = 0;
                                classes[4] = 0;
                                classes[5] = 0;
                            }
                        }
                    } else {
                        if (features[17] <= 1.0217090845108032) {
                            classes[0] = 0;
                            classes[1] = 165;
                            classes[2] = 15;
                            classes[3] = 1029;
                            classes[4] = 0;
                            classes[5] = 0;
                        } else {
                            classes[0] = 1;
                            classes[1] = 129;
                            classes[2] = 4;
                            classes[3] = 88;
                            classes[4] = 0;
                            classes[5] = 0;
                        }
                    }
                } else {
                    if (features[1] <= 1.091892123222351) {
                        if (features[9] <= 0.17686207592487335) {
                            classes[0] = 0;
                            classes[1] = 693;
                            classes[2] = 12;
                            classes[3] = 225;
                            classes[4] = 0;
                            classes[5] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 124;
                            classes[2] = 6;
                            classes[3] = 674;
                            classes[4] = 0;
                            classes[5] = 0;
                        }
                    } else {
                        if (features[1] <= 3.2776291370391846) {
                            if (features[1] <= 1.2781110405921936) {
                                if (features[9] <= 0.1465122401714325) {
                                    classes[0] = 0;
                                    classes[1] = 225;
                                    classes[2] = 5;
                                    classes[3] = 216;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 96;
                                    classes[2] = 26;
                                    classes[3] = 1609;
                                    classes[4] = 0;
                                    classes[5] = 0;
                                }
                            } else {
                                if (features[9] <= 0.43839211761951447) {
                                    classes[0] = 0;
                                    classes[1] = 289;
                                    classes[2] = 116;
                                    classes[3] = 35588;
                                    classes[4] = 15;
                                    classes[5] = 136;
                                } else {
                                    if (features[1] <= 2.5719685554504395) {
                                        classes[0] = 0;
                                        classes[1] = 2;
                                        classes[2] = 0;
                                        classes[3] = 830;
                                        classes[4] = 1;
                                        classes[5] = 31;
                                    } else {
                                        classes[0] = 0;
                                        classes[1] = 1;
                                        classes[2] = 0;
                                        classes[3] = 321;
                                        classes[4] = 3;
                                        classes[5] = 193;
                                    }
                                }
                            }
                        } else {
                            if (features[9] <= 0.3625847399234772) {
                                classes[0] = 0;
                                classes[1] = 3;
                                classes[2] = 0;
                                classes[3] = 606;
                                classes[4] = 1;
                                classes[5] = 173;
                            } else {
                                classes[0] = 0;
                                classes[1] = 0;
                                classes[2] = 0;
                                classes[3] = 81;
                                classes[4] = 1;
                                classes[5] = 173;
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 1.459218978881836) {
                    if (features[17] <= 0.693316251039505) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 15;
                        classes[4] = 1;
                        classes[5] = 72;
                    } else {
                        classes[0] = 17;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 172;
                        classes[4] = 6;
                        classes[5] = 27;
                    }
                } else {
                    if (features[9] <= 0.22058256715536118) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 0;
                        classes[3] = 77;
                        classes[4] = 2;
                        classes[5] = 69;
                    } else {
                        classes[0] = 0;
                        classes[1] = 25;
                        classes[2] = 0;
                        classes[3] = 46;
                        classes[4] = 2;
                        classes[5] = 3861;
                    }
                }
            }
        }

        return findMax(classes);
    };

};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 === 25) {

        // Features:
        var features = process.argv.slice(2);

        // Prediction:
        var pclf = new PrimoDecisionTreeClassifier();
        var prediction = pclf.predict(features);
        console.log(prediction);

    }
}

export default PrimoDecisionTreeClassifier;
