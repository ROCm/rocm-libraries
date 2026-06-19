
#include "header.h"

void predict_unit2(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.310776710510254794) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.551017761230469638) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.0006938946946399022;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.547126770019532138) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.02133506718736724;
                        } else {
                          result[0] += -0.002113970565822251;
                        }
                      } else {
                        result[0] += 0.02554533697715123;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                            result[0] += -0.021764528465554336;
                          } else {
                            result[0] += 0.04213794466370115;
                          }
                        } else {
                          result[0] += -0.0022890873737988478;
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.03336381464911697;
                        } else {
                          result[0] += -0.03189476210042405;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.480630159378052646) ) ) {
                      result[0] += 0.01789721390910081;
                    } else {
                      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.035366357951098916;
                      } else {
                        result[0] += 0.017337594119705393;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.041918491307503955;
                  } else {
                    result[0] += 0.016526331952626653;
                  }
                } else {
                  result[0] += -0.01283078950314241;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.014482790021444737;
              } else {
                result[0] += -0.016470319596402182;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
              result[0] += 0.008174290884896617;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.232423543930054599) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.07895761816088202;
                      } else {
                        result[0] += -0.02829758326436907;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.021139161086487293;
                      } else {
                        result[0] += -0.1301873510464535;
                      }
                    }
                  } else {
                    result[0] += 0.07670183820445292;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                      result[0] += -0.11605166691090468;
                    } else {
                      result[0] += -0.023028746642305836;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0015984431385976444;
                    } else {
                      result[0] += 0.062127880615699564;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.005842363219072787;
                  } else {
                    result[0] += -0.0352987810348379;
                  }
                } else {
                  result[0] += -0.07136735191380998;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0014453317245088032;
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.6739082336425799) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                  result[0] += -0.041144312840612485;
                } else {
                  result[0] += 0.037963449305465885;
                }
              } else {
                result[0] += 0.01211118537370041;
              }
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.93885374069213956) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.011964523368519794;
              } else {
                result[0] += 0.011802967841100223;
              }
            } else {
              result[0] += 0.014012938145965426;
            }
          }
        }
      } else {
        result[0] += -0.004727086257657926;
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.008062917190749481;
      } else {
        result[0] += -0.07021801269927029;
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.69067406654357999) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.04242862199686931;
            } else {
              result[0] += -0.03704776976847634;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.04296633122333596;
            } else {
              result[0] += 0.035174708446398556;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.010694504623208628;
          } else {
            result[0] += -0.008399969024743732;
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += -0.00012527347591632945;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.017126945384596783;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += -0.013934682390837306;
                  } else {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0674247427429417;
                    } else {
                      result[0] += 0.002958862620199608;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.04673803675100387;
                } else {
                  result[0] += 0.04335474686438044;
                }
              }
            }
          } else {
            result[0] += -0.028376577370313107;
          }
        } else {
          result[0] += -0.018914491261715575;
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.055496215820313388) ) ) {
        result[0] += 0.0016784351185098166;
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.003838300704956943) ) ) {
            result[0] += -0.0007860719691851206;
          } else {
            result[0] += -0.01588163874903741;
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.0539720159912852;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.655405282974244052) ) ) {
                  result[0] += -0.019411827817634487;
                } else {
                  result[0] += 0.09242174985457124;
                }
              } else {
                result[0] += 0.03832968016606969;
              }
            } else {
              result[0] += -0.060779461350201784;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0026418333478029805;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.10849876490921505;
              } else {
                result[0] += -0.004983291010037711;
              }
            }
          } else {
            result[0] += 0.01331664229059161;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.5431861877441424) ) ) {
              result[0] += 0.0022917616463010103;
            } else {
              result[0] += -0.06169822159284608;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.241300821304322177) ) ) {
                result[0] += -0.009139034550686945;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
                  result[0] += -0.010263616557991134;
                } else {
                  result[0] += -0.2621395397004773;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                result[0] += -0.28623639808239737;
              } else {
                result[0] += -0.09534053264204811;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
          result[0] += 0.03508355984121527;
        } else {
          result[0] += -0.009290893819592068;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.08859249775497151;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
              result[0] += 0.0923304101806235;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.680161952972413886) ) ) {
                result[0] += -0.14468083129558904;
              } else {
                result[0] += -0.018240646027288455;
              }
            }
          }
        } else {
          result[0] += -0.060398029062909836;
        }
      } else {
        if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                result[0] += -0.03025647376593074;
              } else {
                result[0] += -0.0004688337029371135;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.010593917961009728;
              } else {
                result[0] += 0.01124796675973602;
              }
            }
          } else {
            result[0] += -0.015381791374468656;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                  result[0] += 0.020667164734642945;
                } else {
                  result[0] += 0.00013129796407959613;
                }
              } else {
                result[0] += -0.0003556119020739542;
              }
            } else {
              result[0] += 0.0031782812099834024;
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.44140100479126021) ) ) {
              result[0] += -0.037842501318617744;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.82380008697509943) ) ) {
                  result[0] += -0.017325762555026877;
                } else {
                  result[0] += -0.10899203467260513;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.05350152428273661;
                } else {
                  result[0] += -0.0006626393473305957;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.497206687927246982) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.016996353072464148;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += 0.0003617473749731591;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.052270391317971544;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.020096221555530844;
                  } else {
                    result[0] += -0.06643109576295543;
                  }
                }
              }
            } else {
              result[0] += 0.007091578691079889;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
              result[0] += -0.004102325714962731;
            } else {
              result[0] += 0.026037192859244846;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.006698488097284446;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.04078366672300143;
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.03935746546573178;
                      } else {
                        result[0] += 0.02198458041573501;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                        result[0] += -0.05226638914779028;
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.001064812956231435;
                          } else {
                            result[0] += 0.12507495515163478;
                          }
                        } else {
                          result[0] += -0.033013046513217084;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0005510149285735772;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.012961558474803962;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                          result[0] += -0.02607643358395531;
                        } else {
                          result[0] += 0.012402959912029617;
                        }
                      } else {
                        result[0] += 0.022178992750793682;
                      }
                    } else {
                      result[0] += -0.06545532872131113;
                    }
                  }
                } else {
                  result[0] += 0.03586146147239624;
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.008139563864781483;
                } else {
                  result[0] += -0.03136442120595093;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
            result[0] += 0.00820059143766176;
          } else {
            result[0] += -0.0603419958248557;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += 0.016720690675528657;
          } else {
            result[0] += 0.08206539694209103;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.0073354500610046895;
      } else {
        result[0] += -0.027275828522358953;
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.023858397500279797;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
              result[0] += -0.10869570855727201;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.0920262415084509;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    result[0] += 0.02451418450157319;
                  } else {
                    result[0] += 8.932198166976691e-06;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.035818177235118334;
                  } else {
                    result[0] += 0.0060412655587031225;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.008005987145485553;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.01191141127385931;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.00332988557417911;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.08609117628837333;
                        } else {
                          result[0] += 0.09914899805853816;
                        }
                      } else {
                        result[0] += 0.03651168701227342;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.08665950788776564;
                        } else {
                          result[0] += -0.07683192202055628;
                        }
                      } else {
                        result[0] += -0.012296227560031937;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.009953256474108963;
                    } else {
                      result[0] += 0.10448027116139094;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.014811825977046484;
          } else {
            result[0] += 0.004850605683666562;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.028783076306946365;
      } else {
        result[0] += -0.0002061442647808139;
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.0037653769530488048;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += -0.01852091267753515;
              } else {
                result[0] += -0.09322173946127356;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05479049682617365) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.04993966286126378;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                    result[0] += 0.12764077914609032;
                  } else {
                    result[0] += -0.02764255677610295;
                  }
                }
              } else {
                result[0] += -0.08741015096748914;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
              result[0] += -0.0067254410860259455;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += 0.08042798819193211;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.097527027130127841) ) ) {
                    result[0] += 0.01765180965017148;
                  } else {
                    result[0] += 0.07622499001021812;
                  }
                } else {
                  result[0] += 0.003660594413941015;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.0648887867746558;
            } else {
              result[0] += -0.013789434979790344;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.00906144836956568;
        } else {
          result[0] += -0.0029926621513312133;
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.448499202728272373) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.003150251669975534;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
              result[0] += 0.0056946516842745105;
            } else {
              result[0] += 0.023687297938503892;
            }
          }
        } else {
          result[0] += -0.02269892521612298;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.0010018213291298295;
          } else {
            result[0] += 0.03732027878667573;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                result[0] += -0.04431034795492523;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.012769802751559936;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.947025299072267401) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.03630305705843955;
                      } else {
                        result[0] += -0.06530855622312473;
                      }
                    } else {
                      result[0] += 0.05682102250061867;
                    }
                  } else {
                    result[0] += 0.0792011309558033;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.06744206099887284;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.011522562392667215;
                } else {
                  result[0] += -0.05904966374976848;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01149767873813526;
                } else {
                  result[0] += -0.039240917736590976;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += -0.010053211746695502;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                        result[0] += 0.0069682279001713684;
                      } else {
                        result[0] += -0.10097213534526023;
                      }
                    } else {
                      result[0] += 0.022416429262063678;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.471622467041016513) ) ) {
                    result[0] += 0.02546389777955977;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                      result[0] += -0.05987799100130731;
                    } else {
                      result[0] += 0.04849242011172063;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.09943054609595678;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.497206687927246982) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
              result[0] += -0.0007602945378058954;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                  result[0] += 0.022286666390019205;
                } else {
                  result[0] += -0.018108036288724923;
                }
              } else {
                result[0] += 0.01372986916347875;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.05675232564497728;
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.509355545043946201) ) ) {
                  result[0] += -0.029281760879534935;
                } else {
                  result[0] += 0.006659687921767442;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                    result[0] += 0.020812519796853588;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                      result[0] += -0.07317507591171961;
                    } else {
                      result[0] += -0.008837071004828286;
                    }
                  }
                } else {
                  result[0] += -0.04407105006597854;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
                result[0] += 0.013313639102234204;
              } else {
                result[0] += 0.1004648208117792;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.0013042256344459391;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                  result[0] += -0.015800817569516703;
                } else {
                  result[0] += 0.035144700822992675;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                  result[0] += 0.021053302478385872;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                    result[0] += -0.0025246575616916993;
                  } else {
                    result[0] += 0.07272845171055131;
                  }
                }
              } else {
                result[0] += -0.03784190130759077;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.019424204945278807;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
                      result[0] += -0.028377870869711594;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.03478326862089494;
                        } else {
                          result[0] += -0.012199954957762963;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                          result[0] += 0.004842204036965901;
                        } else {
                          result[0] += -0.10791806470132928;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.000660946564119279;
                  } else {
                    result[0] += -0.014293264728800888;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                  result[0] += 0.005248971228721382;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.2767210006713885) ) ) {
                          result[0] += 0.010670439591113263;
                        } else {
                          result[0] += 0.030993672830555243;
                        }
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.139882326126099521) ) ) {
                          result[0] += 0.009525501383766361;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40908622741699396) ) ) {
                            result[0] += 0.02799467333673508;
                          } else {
                            result[0] += -0.05807137881301658;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.02206516840861721;
                          } else {
                            result[0] += 0.012800566086951574;
                          }
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += 0.05133403532390997;
                              } else {
                                result[0] += 0.1482067973872337;
                              }
                            } else {
                              result[0] += -0.005962359196365905;
                            }
                          } else {
                            result[0] += -0.0009989481067652357;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.07617484993398034;
                        } else {
                          result[0] += -0.0558745990125995;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.0012432211587887578;
                      } else {
                        result[0] += -0.11699831409710049;
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.035460111384335566;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.009205084357273364;
                        } else {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                            result[0] += -0.03739403228418019;
                          } else {
                            result[0] += 0.1492372951400842;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.07817801963457494;
              } else {
                result[0] += 0.017435567949372257;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.07011683841387532;
                } else {
                  result[0] += -0.01767634629078425;
                }
              } else {
                result[0] += 0.0777071410254514;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.092464203738091;
                } else {
                  result[0] += -0.0005592122829439799;
                }
              } else {
                result[0] += -0.020238986101802452;
              }
            } else {
              result[0] += -0.009916274202737836;
            }
          }
        } else {
          result[0] += 0.022070482342255317;
        }
      }
    } else {
      result[0] += -0.009813579350099707;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.61744737625122248) ) ) {
      result[0] += -0.00034658224436677536;
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.0009563776622068667;
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
          result[0] += -0.01777378833356889;
        } else {
          result[0] += 0.07290681103000747;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
          result[0] += -0.004619472854405063;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.009344216235580292;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                result[0] += -0.0310348810553214;
              } else {
                result[0] += -0.09417960316408876;
              }
            }
          } else {
            result[0] += -0.0725478282629545;
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.021093470912128987;
        } else {
          result[0] += -0.008069608531864935;
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
          result[0] += 0.0008844896729164859;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.01262459627101139;
          } else {
            result[0] += 0.011029704715998817;
          }
        }
      } else {
        result[0] += 0.0013596696260387226;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += 0.0008217895927773346;
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.002379886457973147;
          } else {
            result[0] += -0.03737458407943201;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.00411602554414763;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.057297745321001264;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.029062807807430487;
                  } else {
                    result[0] += -0.016055497622451273;
                  }
                } else {
                  result[0] += 0.02579557133833011;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.004403471366522061;
                  } else {
                    result[0] += 0.04432430663226425;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.0031863175320197197;
                    } else {
                      result[0] += -0.07243710075632913;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.022251456907329287;
                      } else {
                        result[0] += -0.044656601635586456;
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0134040947061572;
                      } else {
                        result[0] += -0.05743188995300013;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.004831694702456168;
                      } else {
                        result[0] += -0.06957508916924564;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.009884849817218325;
                        } else {
                          result[0] += -0.03339291442724895;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                          result[0] += 0.20743665262227;
                        } else {
                          result[0] += -0.0014968880224132163;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.0028424536312239637;
                        } else {
                          result[0] += 0.06537136322030421;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
                          result[0] += 0.06948731049447499;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                            result[0] += 0.04057073418273408;
                          } else {
                            result[0] += -0.023641704120175817;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.0010624705200614688;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                          result[0] += -0.16503969717609357;
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.06367286223405126;
                          } else {
                            result[0] += -0.016960597150467462;
                          }
                        }
                      } else {
                        result[0] += 0.07540058724321026;
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.04051564503291113;
                      } else {
                        result[0] += 0.11613555853460629;
                      }
                    }
                  } else {
                    result[0] += -0.05206705237700043;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += 0.033321001229937565;
                  } else {
                    result[0] += -0.02358744682936351;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.0912622586060692;
                      } else {
                        result[0] += -0.014391004510292717;
                      }
                    } else {
                      result[0] += 0.004930629901607866;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
                        result[0] += 0.020038298952152252;
                      } else {
                        result[0] += 0.0678161927046555;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
                          result[0] += 0.01693651812631569;
                        } else {
                          result[0] += -0.007936779418308906;
                        }
                      } else {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.022093280939735067;
                        } else {
                          result[0] += 0.05141580479933155;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.164715528488160068) ) ) {
                    result[0] += 0.02873303587167584;
                  } else {
                    result[0] += -0.015623878650992655;
                  }
                } else {
                  result[0] += -0.026885480179246853;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
            result[0] += -0.011605478372019496;
          } else {
            result[0] += -0.04271842553102534;
          }
        }
      }
    } else {
      result[0] += -0.0179227418949315;
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
      result[0] += -0.009541697325945633;
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.29409265518188654) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
                  result[0] += 0.007383971730544461;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.007308789424146808;
                  } else {
                    result[0] += -0.1224279605133333;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0024578179455955894;
                    } else {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.12023778159262928;
                      } else {
                        result[0] += -0.0442419263033307;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.0027804536302276726;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.019086213565893403;
                      } else {
                        result[0] += 0.05007685160715658;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                    result[0] += -0.005013136739923511;
                  } else {
                    result[0] += 0.0016964114222987458;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += -0.009598619169503005;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.07469755098036517;
                        } else {
                          result[0] += 0.07670735677924967;
                        }
                      } else {
                        result[0] += -0.02701351958175128;
                      }
                    }
                  } else {
                    result[0] += -0.004045244151787349;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                    result[0] += -0.12437030421260226;
                  } else {
                    result[0] += 0.04842209794503686;
                  }
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                    result[0] += 0.014686098804511423;
                  } else {
                    result[0] += -0.014535339068455855;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += 0.012552259827456218;
                  } else {
                    result[0] += -0.03755009010337079;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.005802579631039184;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.02367217345272392;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.628555774688722479) ) ) {
                      result[0] += -0.08779506125126689;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.04952343384048958;
                      } else {
                        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.0663042944711502;
                        } else {
                          result[0] += 0.08129630080435973;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.0456267728671699;
                }
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.05017687533539692;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.650573849678039995) ) ) {
                  result[0] += 0.007209384572635586;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += -0.05219719392819425;
                  } else {
                    result[0] += 0.07604627175881905;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.06363845393129516;
              } else {
                result[0] += -0.03582483861423116;
              }
            } else {
              result[0] += -0.023585038054358874;
            }
          } else {
            result[0] += 0.015930408964950488;
          }
        }
      } else {
        result[0] += -0.008537459344262465;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.158509254455567294) ) ) {
        result[0] += 0.0004359833556428771;
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.014412875083727426;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
                result[0] += 0.020636441200534164;
              } else {
                result[0] += -0.04063590509244174;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += -0.00017630715513831965;
            } else {
              result[0] += -0.015608359221581875;
            }
          }
        } else {
          result[0] += -0.027287106633362524;
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.970085620880127397) ) ) {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.021518211124228793;
            } else {
              result[0] += -0.04641756311218373;
            }
          } else {
            result[0] += -0.010027606536045154;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.008692709020058835;
            } else {
              result[0] += -0.050559047051836836;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += -0.012316054503452883;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03148198719842711;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.02174202020100749;
                  } else {
                    result[0] += 0.02313999666680845;
                  }
                } else {
                  result[0] += 0.017240995266701097;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                result[0] += -0.004522506976525768;
              } else {
                result[0] += -0.016386342890251662;
              }
            } else {
              result[0] += -0.04680538271014285;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += -0.004890354365840443;
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.028158783658453514;
              } else {
                result[0] += 0.06152470095359136;
              }
            }
          }
        } else {
          result[0] += 0.02120650873204541;
        }
      }
    }
  }
  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.198252916336060458) ) ) {
      result[0] += 0.0016112153440325;
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.07352271519176255;
          } else {
            result[0] += 0.04207996144787973;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.027203839706023226;
            } else {
              result[0] += 0.04664403880653986;
            }
          } else {
            result[0] += -0.04852528501797157;
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
          result[0] += 0.00019709893217492106;
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += 0.005819600908456981;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.005480044955266515;
                    } else {
                      result[0] += -0.10669249011804574;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      result[0] += 0.07696252535516201;
                    } else {
                      result[0] += -0.04043342783024465;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.022316169992311785;
                } else {
                  result[0] += -0.00042984099869757024;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.08465554012281844;
                } else {
                  result[0] += -0.08160766230303156;
                }
              } else {
                result[0] += 0.06711034447805571;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.004014566977951508;
                } else {
                  result[0] += 0.008246274414797876;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.004091503265370218;
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.07301482317893562;
                  } else {
                    result[0] += -0.017716450574129083;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.551017761230469638) ) ) {
                        result[0] += -0.11838395959056235;
                      } else {
                        result[0] += -0.032752226793980745;
                      }
                    } else {
                      result[0] += -0.014883000666716207;
                    }
                  } else {
                    result[0] += -0.06872773404967085;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.0022814439604564767;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.07873835863041753;
                    } else {
                      result[0] += 0.00739708601792483;
                    }
                  }
                }
              } else {
                result[0] += -0.046285176689627;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += 4.5093093955428666e-05;
      } else {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.007048744997695535;
        } else {
          result[0] += 0.03346058645214391;
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
          result[0] += 0.004175524391254029;
        } else {
          result[0] += -0.006165445574517078;
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.417592287063599077) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                result[0] += 0.03732044661670142;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += -0.03331023288543211;
                  } else {
                    result[0] += -0.129057946317036;
                  }
                } else {
                  result[0] += -0.0032146815712159717;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                result[0] += -0.03531299638593489;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.947025299072267401) ) ) {
                    result[0] += -0.06215044574820963;
                  } else {
                    result[0] += 0.062276146111065404;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.08779247001624028;
                  } else {
                    result[0] += -0.04939227409016461;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0006902737032774404;
            } else {
              result[0] += 0.006452142131340287;
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.027116603359172004;
                    } else {
                      result[0] += 0.009993849311173635;
                    }
                  } else {
                    result[0] += 0.03941918981430792;
                  }
                } else {
                  result[0] += 0.04740159460198469;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                  result[0] += 0.02788316582289446;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.00566391551415329;
                  } else {
                    result[0] += -0.04619364881095928;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.07899014497332885;
                  } else {
                    result[0] += 0.016885214945286886;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.09275333428824056;
                  } else {
                    result[0] += 0.030877712013616;
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0005235872329846439;
                    } else {
                      result[0] += 0.039366249913413136;
                    }
                  } else {
                    result[0] += -0.002387327814177433;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += -0.06282268916528522;
                  } else {
                    result[0] += 0.030861047220985434;
                  }
                }
              }
            }
          } else {
            result[0] += -0.015305529136109991;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
      result[0] += -0.009283718477763632;
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.29409265518188654) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
              result[0] += 0.0025540662180948874;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.006067829137060027;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0663271319544833;
                } else {
                  result[0] += -0.01679048051023216;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.007174343911391691;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.0111615694223106;
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
                              result[0] += -0.03977795064812885;
                            } else {
                              result[0] += 0.0919991859173673;
                            }
                          } else {
                            result[0] += -0.08826772481796313;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                        result[0] += -0.012384047075213579;
                      } else {
                        result[0] += 0.029030223037843333;
                      }
                    }
                  } else {
                    result[0] += -0.0011346885935155213;
                  }
                } else {
                  result[0] += 0.026726756138256202;
                }
              } else {
                result[0] += -0.02315832508940521;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.04507624782339363;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
                          result[0] += 0.017657488401712003;
                        } else {
                          result[0] += -0.0032344860777643227;
                        }
                      } else {
                        result[0] += -0.02886115017253755;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.025676258454461084;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.381086945533752885) ) ) {
                          result[0] += -0.03635947207596082;
                        } else {
                          result[0] += 0.023225331880942755;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0067029470487592144;
                    } else {
                      result[0] += 0.00415380221786207;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.04449613661499696;
                      } else {
                        result[0] += -0.0008916429065486476;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        result[0] += 0.012265565865311481;
                      } else {
                        result[0] += -0.04228578754379337;
                      }
                    }
                  } else {
                    result[0] += 0.009857350801757151;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += 0.012460955546996678;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.12391448038932816;
                        } else {
                          result[0] += -0.045470205294646855;
                        }
                      }
                    } else {
                      result[0] += 0.0114795082979338;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
                      result[0] += 0.013266059773772066;
                    } else {
                      result[0] += 0.048555398037849605;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.01155970759279655;
        }
      } else {
        result[0] += -0.007851447707168687;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 5.476559799550314e-05;
        } else {
          result[0] += -0.013055469607624527;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.007688330456741928;
        } else {
          result[0] += -0.0379111394607814;
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += 0.009463245686751155;
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.007346989010664599;
              } else {
                result[0] += -0.06490222651056146;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.03979867216097226;
              } else {
                result[0] += 0.007385047358370056;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.003763637477478583;
              } else {
                result[0] += -0.019162768606314197;
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += -0.012156794831680744;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.05070498871148358;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.008013011998587923;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                          result[0] += 0.030262651596911223;
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.0018250442437913947;
                          } else {
                            result[0] += 0.10883331144774043;
                          }
                        }
                      } else {
                        result[0] += -0.019532480597305554;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.025881597023277022;
                  } else {
                    result[0] += -0.02556439082959108;
                  }
                } else {
                  result[0] += 0.008518573536389273;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0468470939175557;
            } else {
              result[0] += 0.08343267504538394;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
            result[0] += -0.013775844714467826;
          } else {
            result[0] += -0.036812659392175436;
          }
        } else {
          result[0] += 0.004343332944161083;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
      result[0] += -0.008726442694933745;
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.25333833694458185) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.799905776977539951) ) ) {
                    result[0] += 0.006024680296378773;
                  } else {
                    result[0] += 0.018230573772510444;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.006843760728673311;
                  } else {
                    result[0] += -0.11262908009493784;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0024677109153761183;
                    } else {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.11217004156869309;
                      } else {
                        result[0] += -0.043735572397631;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += -0.16303664024242837;
                        } else {
                          result[0] += -0.021168990082804116;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                          result[0] += 0.1013642937180898;
                        } else {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            result[0] += 0.016400417024065948;
                          } else {
                            result[0] += -0.03478735430837331;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.03305604503638274;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                    result[0] += -0.0047478536665047436;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
                      result[0] += 0.002928579041106157;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.020441985857241593;
                      } else {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.11247142616629605;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                              result[0] += 0.011770540167647364;
                            } else {
                              result[0] += -0.034507450317248965;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.561121463775635654) ) ) {
                              result[0] += -0.047681939126419964;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                                result[0] += -0.01291674921409941;
                              } else {
                                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                                  result[0] += 0.1027587963287242;
                                } else {
                                  result[0] += 0.03309394770034204;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.04830298486957375;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.12583853378151227;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.07268381118774592) ) ) {
                        result[0] += 0.07026135331548339;
                      } else {
                        result[0] += -0.06776686211248391;
                      }
                    } else {
                      result[0] += -0.000621501859369151;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.03244696964827311;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.285166740417482245) ) ) {
                            result[0] += -0.012082439352993517;
                          } else {
                            result[0] += 0.13488339042320904;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                            result[0] += 0.07739410834330016;
                          } else {
                            result[0] += -0.07296192903898956;
                          }
                        }
                      } else {
                        result[0] += 0.012930578734587651;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.08564112530064355;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.09572848500316539;
                        } else {
                          result[0] += -0.03254841720375446;
                        }
                      }
                    } else {
                      result[0] += 0.0011431905403792924;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += -0.002695793294431394;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.88338184356689631) ) ) {
                      result[0] += -0.05345157221487954;
                    } else {
                      result[0] += -0.0019483917057888961;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.002839348565806724;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.06593676865275581;
                    } else {
                      result[0] += 0.01511736930101352;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.007586887182099886;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                    result[0] += 0.029734160164613874;
                  } else {
                    result[0] += -0.026865503469526902;
                  }
                } else {
                  result[0] += 0.03358376685086314;
                }
              }
            } else {
              result[0] += -0.013147996947766563;
            }
          }
        } else {
          result[0] += 0.016778757395825652;
        }
      } else {
        result[0] += -0.007591935428564208;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      result[0] += -0.0002696793409338985;
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -2.3607899960300787e-05;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.02215178951931758;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.047747470032010506;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.97393989562988459) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
                  result[0] += 0.014199208452897417;
                } else {
                  result[0] += -0.003417417520259996;
                }
              } else {
                result[0] += -0.02530487410872468;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.0016153018653828543;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
              result[0] += -0.010142221832071585;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += -0.0426728753806938;
              } else {
                result[0] += -0.019195707268877208;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
          result[0] += 0.06002503905698915;
        } else {
          result[0] += 0.013989511871686754;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.123302459716797763) ) ) {
            result[0] += 0.01811141581514948;
          } else {
            result[0] += 0.18477910310256948;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.90896487236023038) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += 0.006493168835457196;
              } else {
                result[0] += -0.049120274918021324;
              }
            } else {
              result[0] += 0.008203146348494844;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
              result[0] += 0.0051940450199677345;
            } else {
              result[0] += -0.02022534782962282;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += -0.0001758575701731259;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.12156401654061447;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += 0.1295767723209564;
                  } else {
                    result[0] += -0.11817544250613535;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.40940880775451749) ) ) {
                  result[0] += -0.008928396312027633;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.16021328753583858;
                    } else {
                      result[0] += 0.002805256666681482;
                    }
                  } else {
                    result[0] += 0.1735271421528297;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.00764274597168146) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                result[0] += 0.0825178618814936;
              } else {
                result[0] += -0.031389050898029694;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.03340738573150282;
              } else {
                result[0] += 0.011511982030659805;
              }
            }
          }
        } else {
          result[0] += -0.029411299474494997;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.023410452695762293;
              } else {
                result[0] += -0.001971754444530538;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                  result[0] += -0.028728859124670127;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += -0.04108955513895673;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.407877445220948154) ) ) {
                        result[0] += 0.02097607934310352;
                      } else {
                        result[0] += 0.08283135769533703;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                        result[0] += -0.011512936710339159;
                      } else {
                        result[0] += 0.039526069218528255;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.056725911498418026;
              }
            }
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.05576916105185941;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                        result[0] += 0.0002636873427241371;
                      } else {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.11340765679871392;
                          } else {
                            result[0] += -0.05255294104928602;
                          }
                        } else {
                          result[0] += -0.0007325218288751767;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.3669629096984881) ) ) {
                      result[0] += -0.014471583085685458;
                    } else {
                      result[0] += -0.050135521254846785;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
                      result[0] += 0.0010337467377693334;
                    } else {
                      result[0] += -0.018270631130907773;
                    }
                  } else {
                    result[0] += 0.007523574517240389;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.00790517389696202;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                      result[0] += -0.0034931845805311113;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.187374591827394354) ) ) {
                          result[0] += 0.00334192368272645;
                        } else {
                          result[0] += 0.029816531453187484;
                        }
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          result[0] += 0.04997224285803037;
                        } else {
                          result[0] += 0.004681894752387676;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.02040683684195877;
                  }
                }
              }
            } else {
              result[0] += -0.003622600958475889;
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.002114278505454019;
          } else {
            result[0] += 0.014486173797391078;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
        result[0] += 0.0001969683145576445;
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.0030680121003621406;
        } else {
          result[0] += -0.02554788834737462;
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.0021991727499818797;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += 0.054074431603406185;
              } else {
                result[0] += -0.008999080372823235;
              }
            } else {
              result[0] += -0.02811716819766648;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.03096372855233407;
              } else {
                result[0] += -0.007912398309922596;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.02925822334278528;
              } else {
                result[0] += 0.04468100591845355;
              }
            }
          }
        }
      } else {
        result[0] += -0.020175709158337916;
      }
    }
  }
  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
      result[0] += 0.18457599143726916;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        result[0] += 0.022405152161715042;
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
          result[0] += 0.03396518879646492;
        } else {
          result[0] += -0.17290807989203574;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.228399038314820224) ) ) {
            result[0] += -0.00753031548581956;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.136462926864624912) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.0052622273800504195;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += 0.012227473231475501;
                      } else {
                        result[0] += -0.10856910088924424;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.006106796051009032;
                          } else {
                            result[0] += -0.13826969554503216;
                          }
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                                result[0] += -0.02171783102588572;
                              } else {
                                result[0] += 0.017484970034357995;
                              }
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += 0.009481321313752084;
                                } else {
                                  result[0] += -0.02978738811335148;
                                }
                              } else {
                                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                                  result[0] += 0.0007423020788239549;
                                } else {
                                  result[0] += 0.03222061589777523;
                                }
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.01888143056128215;
                            } else {
                              result[0] += -0.021028478059432695;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.024606939618162082;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.005961643050860786;
                    } else {
                      result[0] += -0.029073527202033447;
                    }
                  }
                } else {
                  result[0] += 0.000785312250172251;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.01176604225597717;
                } else {
                  result[0] += -0.0011737114319405467;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
                    result[0] += -0.004332339423875057;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.478159427642823154) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35311269760132014) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.422362327575684482) ) ) {
                            result[0] += -0.022494733209578302;
                          } else {
                            result[0] += -0.11280349580362158;
                          }
                        } else {
                          result[0] += 0.013680263257603065;
                        }
                      } else {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.1287067279722101;
                        } else {
                          result[0] += 0.01627622789998524;
                        }
                      }
                    } else {
                      result[0] += 0.03781259109343419;
                    }
                  }
                } else {
                  result[0] += 0.08458272160035228;
                }
              }
            }
          }
        } else {
          result[0] += -0.01320400510540169;
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.010366891455205845;
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40908622741699396) ) ) {
              result[0] += -0.0024665805487009453;
            } else {
              result[0] += 0.011126326018966773;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.688684463500978339) ) ) {
              result[0] += 0.03212387570860291;
            } else {
              result[0] += 0.006056963417408615;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
        result[0] += -0.0003916805653815108;
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.970085620880127397) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.031809302512032615;
                      } else {
                        result[0] += 0.06465656188628587;
                      }
                    } else {
                      result[0] += -0.010595315445427318;
                    }
                  } else {
                    result[0] += 0.0637349293589309;
                  }
                } else {
                  result[0] += -0.030691365204693833;
                }
              } else {
                result[0] += -0.008002385497581725;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.01889719683147557;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.011143829231975358;
                } else {
                  result[0] += 0.01134490407828685;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                  result[0] += 0.06022509548063783;
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.010477328700407186;
                  } else {
                    result[0] += -0.015334225733003646;
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                      result[0] += -0.0434508945760464;
                    } else {
                      result[0] += -0.010215919815105307;
                    }
                  } else {
                    result[0] += -0.03758936561693663;
                  }
                } else {
                  result[0] += 0.007927235328917532;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.005130850888288538;
                } else {
                  result[0] += -0.025017972836515714;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                    result[0] += -0.011770030420562044;
                  } else {
                    result[0] += 0.08684858268888668;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                    result[0] += 0.009583578324546283;
                  } else {
                    result[0] += 0.04327014552845074;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.01097472764235595;
          } else {
            result[0] += -0.04746685759764827;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
      result[0] += 0.18471270151766683;
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
          result[0] += 0.017620555406563737;
        } else {
          result[0] += 0.07504205774828414;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
          result[0] += 0.012155246589711349;
        } else {
          result[0] += -0.13665153156211451;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
            result[0] += -0.0046817378999172675;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.13282456965207703;
            } else {
              result[0] += -0.03323424467908794;
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.021407851686850915;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                    result[0] += -0.10376535996995795;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.08975976121666548;
                    } else {
                      result[0] += 0.003713298716276641;
                    }
                  }
                } else {
                  result[0] += -0.019414169405135753;
                }
              } else {
                result[0] += 0.0023683791123657576;
              }
            }
          } else {
            result[0] += -0.0001458804583443362;
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
            result[0] += 0.060460490702535334;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.005713281667089095;
            } else {
              result[0] += -0.1116910361211099;
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                result[0] += 0.0010598181648496616;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                  result[0] += 0.061467959966530344;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.92430353164673029) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.957021474838257724) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.711739778518677646) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.698757410049439365) ) ) {
                            result[0] += 0.04093188932292252;
                          } else {
                            result[0] += -0.09966050310988807;
                          }
                        } else {
                          result[0] += -0.09611704580879118;
                        }
                      } else {
                        result[0] += 0.07978118625261811;
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.579273939132691318) ) ) {
                            result[0] += 0.001667035291929294;
                          } else {
                            result[0] += -0.14929599846511002;
                          }
                        } else {
                          result[0] += 0.014768579192654153;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                          result[0] += 0.033252385734271854;
                        } else {
                          result[0] += -0.017323818847621342;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.007622352302785192;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                result[0] += -0.0065800705659836275;
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.053948907819684956;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += 0.05343265681974142;
                  } else {
                    result[0] += -0.01888752415073475;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.0008678771330513032;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.58961367607116788) ) ) {
                    result[0] += 0.08603997637462829;
                  } else {
                    result[0] += -0.004025454756908631;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.132848501205445224) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += 0.08321532915791341;
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.04830097239552377;
                      } else {
                        result[0] += 0.04281102698708236;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.226934909820557529) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += -0.05776369764840039;
                      } else {
                        result[0] += -0.1531626626818857;
                      }
                    } else {
                      result[0] += 0.03988853984555282;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                        result[0] += -0.12271333401963654;
                      } else {
                        result[0] += 0.07925431072789263;
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.16396397311869734;
                      } else {
                        result[0] += 0.010015583017372742;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.027608822229278837;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                          result[0] += 0.0009290954510283627;
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.05854714498547273;
                          } else {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.543220520019532138) ) ) {
                              result[0] += 0.03171967128526025;
                            } else {
                              result[0] += -0.021028136008151083;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                          result[0] += -0.05613405767122662;
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.83939445018768355) ) ) {
                            result[0] += 0.030246018296488625;
                          } else {
                            result[0] += -0.01908226991852148;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += -0.09854748690691377;
                  } else {
                    result[0] += -0.018459313980612648;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += -0.07864623457852792;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
          result[0] += -0.049863965240823586;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.03474080516225826;
              } else {
                result[0] += -0.0025634658046147815;
              }
            } else {
              result[0] += -0.029741851662138247;
            }
          } else {
            result[0] += -0.03658428621746745;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
      result[0] += 0.1848333568105068;
    } else {
      result[0] += 0.01685817517495398;
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.009543354193049173;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.02201395262535783;
                    } else {
                      result[0] += -0.03666448515448818;
                    }
                  } else {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                            result[0] += -0.08139279419234835;
                          } else {
                            result[0] += 0.01576691516761854;
                          }
                        } else {
                          result[0] += 0.03393885252275929;
                        }
                      } else {
                        result[0] += -0.007290341194862865;
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.03243800196548294;
                      } else {
                        result[0] += -0.008240016464091657;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.954540252685547763) ) ) {
                  result[0] += -0.0011159806774112676;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.007870463192305813;
                  } else {
                    result[0] += 0.019921430512940642;
                  }
                }
              }
            } else {
              result[0] += 0.011721849184903385;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                  result[0] += 0.026093285811565323;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.029531917787449576;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.0005216335974490992;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.028379144625174813;
                      } else {
                        result[0] += -0.0008769161826953351;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0552439528131635;
              }
            } else {
              result[0] += 0.002487005420058802;
            }
          }
        } else {
          result[0] += -0.012615785893723575;
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.01116580782466049;
          } else {
            result[0] += 0.12374369119207972;
          }
        } else {
          result[0] += 0.00862649467490563;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
            result[0] += -0.01094803334782327;
          } else {
            result[0] += 0.023292839773319605;
          }
        } else {
          result[0] += -0.015100068882181641;
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.007754123446895936;
                } else {
                  result[0] += 0.013162905430709885;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.119963221086966;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.005934724418152498;
                  } else {
                    result[0] += -0.04051786597006826;
                  }
                }
              }
            } else {
              result[0] += -0.0368091039682015;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.015003352857027969;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.149111986160279208) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.006686652139321916;
                    } else {
                      result[0] += 0.0508920312880274;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.667095184326172763) ) ) {
                      result[0] += 0.02300287961421429;
                    } else {
                      result[0] += -0.04749051970574589;
                    }
                  }
                } else {
                  result[0] += 0.027651457176169687;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += 0.022215228877790413;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += 0.0006794425337497687;
                    } else {
                      result[0] += -0.07355403408233059;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                    result[0] += -0.015260626742722567;
                  } else {
                    result[0] += 0.027730499988321784;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.006471365259706034;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.03095142271498949;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.03064138767643794;
                    } else {
                      result[0] += 0.04859426834038164;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.008241658703293717;
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                          result[0] += -0.02405266428118211;
                        } else {
                          result[0] += 0.013394205277364535;
                        }
                      } else {
                        result[0] += -0.03368977944563811;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                        result[0] += -0.018061034799767537;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.27828097343444913) ) ) {
                          result[0] += -0.0012810382049138313;
                        } else {
                          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.05996798787150375;
                          } else {
                            result[0] += 0.009242060910574208;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.046109842252058095;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.276966691017151323) ) ) {
                        result[0] += 0.005223896770104285;
                      } else {
                        result[0] += -0.028342708797880957;
                      }
                    } else {
                      result[0] += -0.06427536895438787;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 2.9257182558387357e-05;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.531673669815064365) ) ) {
      result[0] += 0.014991620234565373;
    } else {
      result[0] += 0.18501349148309315;
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.003770296079785646;
              } else {
                result[0] += -0.01929496645635865;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.10607491881709392;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                        result[0] += 0.026027942808068007;
                      } else {
                        result[0] += 0.16717648989714007;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.16422008284619294;
                    } else {
                      result[0] += 0.01361967738684939;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.012884362102652683;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.436733961105347568) ) ) {
                        result[0] += -0.017169269590333488;
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.0887893981572585;
                        } else {
                          result[0] += 0.015546949676443206;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.198252916336060458) ) ) {
                        result[0] += 0.040842113851547526;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                          result[0] += 0.06821908370972724;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                            result[0] += 0.09783927945159494;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                              result[0] += -0.07791119763646408;
                            } else {
                              result[0] += 0.0020398726430250597;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.418550252914429599) ) ) {
                  result[0] += -0.006632994016035726;
                } else {
                  result[0] += 0.001021975992122245;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += 0.07795975559666603;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.00030773004992637904;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += 0.008713302397264233;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.021007238576510424;
                    } else {
                      result[0] += 0.0819071681950457;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.00946668004586007;
              } else {
                if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.026205205577316728;
                } else {
                  result[0] += -0.009195882919594642;
                }
              }
            }
          }
        } else {
          result[0] += -0.011475799970250502;
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.005280278723539144;
            } else {
              result[0] += -0.16067639788940152;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.714014530181885654) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  result[0] += -0.11592676549013936;
                } else {
                  result[0] += 0.028823937520783577;
                }
              } else {
                result[0] += -0.19349060567103912;
              }
            } else {
              result[0] += 0.09344358238364733;
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += 0.039966666658093036;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.050540946892317695;
              } else {
                result[0] += 0.0004658609395730541;
              }
            }
          } else {
            result[0] += 0.01608194173782073;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.570234775543213779) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.055777966178534144;
            } else {
              result[0] += 0.0014348741208939683;
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += 0.0013196271797873632;
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.02001344319603643;
                      } else {
                        result[0] += 0.004909135012098518;
                      }
                    } else {
                      result[0] += 0.01057085389634761;
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
                      result[0] += 0.037664652578594235;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.353499650955201083) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                            result[0] += 0.0066152993121093035;
                          } else {
                            result[0] += -0.032782982570329904;
                          }
                        } else {
                          result[0] += -0.029993842555945983;
                        }
                      } else {
                        result[0] += -0.05313831055283885;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0032132166526539774;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += 0.052740907286958995;
                    } else {
                      result[0] += -0.007742567298999948;
                    }
                  } else {
                    result[0] += 0.0067750587777784385;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.006805050007622083;
                  } else {
                    result[0] += -0.00286864182410826;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += -0.022837981898843843;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                    result[0] += 0.006691264466916306;
                  } else {
                    result[0] += -0.009093283986423428;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.012860581076598217;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.03721036468107526;
          } else {
            result[0] += 0.040994527592136926;
          }
        } else {
          result[0] += 0.0004937990225846156;
        }
      }
    }
  }
  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += 0.001951879468348414;
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += 0.01786872386576818;
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.004205199400258793;
                } else {
                  result[0] += -0.018757070060116076;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.665476083755494052) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.014999941556940967;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.774546623229981357) ) ) {
                        result[0] += 0.09715916505043846;
                      } else {
                        result[0] += 0.011807668261929236;
                      }
                    }
                  } else {
                    result[0] += -0.0386323740943702;
                  }
                } else {
                  result[0] += -0.05264716319279361;
                }
              }
            }
          } else {
            result[0] += 0.0035311860501537036;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0006895561781600068;
              } else {
                result[0] += 0.018643637132095044;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.10928304654955122;
                  } else {
                    result[0] += -0.04658480026910092;
                  }
                } else {
                  result[0] += 0.011315025405379016;
                }
              } else {
                result[0] += 0.04369264602950866;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.08049045691842394;
              } else {
                result[0] += -0.006065553404042624;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += 0.006233147067472378;
              } else {
                result[0] += 0.1222069875363555;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.018785255289838907;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.06826680799487733;
                } else {
                  result[0] += 0.005843720862640797;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.01523831002540569;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += 0.021801065625313584;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07907948492939873;
                  } else {
                    result[0] += 0.03889535753795553;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.777674019336700661) ) ) {
                      result[0] += 0.002560960139936565;
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.03360145387258531;
                      } else {
                        result[0] += -0.08952575968096213;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.012977685712326015;
                    } else {
                      result[0] += 0.002154057888067401;
                    }
                  }
                } else {
                  result[0] += -0.04937887260240837;
                }
              } else {
                result[0] += -0.054392967266967907;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.0744494207423979;
              } else {
                result[0] += -0.01913615578726941;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.011390974999172984;
    }
  } else {
    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.0003825545204890117;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.799905776977539951) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.012382947426093364;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
              result[0] += 0.02893080533335537;
            } else {
              result[0] += -0.03145475631130018;
            }
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.663770914077759677) ) ) {
            result[0] += -0.0528718995762506;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.04536194006707192;
            } else {
              result[0] += 0.023104358277291525;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.00793971442452048;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.777674019336700661) ) ) {
                  result[0] += 0.07007568142712516;
                } else {
                  result[0] += -0.0006071921906674345;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03774080132277566;
                } else {
                  result[0] += 0.009071412692419926;
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.0051914187809986466;
              } else {
                result[0] += -0.03163613283325293;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.774546623229981357) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.061553560894844664;
                } else {
                  result[0] += 0.009416331373058527;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0007862745912251576;
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.07435233648101118;
                    } else {
                      result[0] += 0.008066286576010962;
                    }
                  }
                } else {
                  result[0] += 0.05310848614908406;
                }
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.27828097343444913) ) ) {
                  result[0] += 0.023480794274835362;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.004884411382473437;
                  } else {
                    result[0] += 0.10697021300783316;
                  }
                }
              } else {
                result[0] += -0.00582063057091818;
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.08461998648397814;
            } else {
              result[0] += 0.02044665552920806;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
        result[0] += 0.0016019228999219711;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += 0.003963097874620157;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.0007770083973256624;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.0035437483383238925;
                      } else {
                        result[0] += 0.10147809678710598;
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.03903977248350914;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                            result[0] += -0.03769948553172453;
                          } else {
                            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.012060459122586228;
                            } else {
                              result[0] += 0.02524675477361801;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.06199269605362828;
                      }
                    }
                  } else {
                    result[0] += 0.0012089061357037458;
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.21020567962095746;
                  } else {
                    result[0] += -0.05155844580076352;
                  }
                }
              } else {
                result[0] += 0.0016773734895315345;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.01130043214692143;
            } else {
              result[0] += -0.010425004992554186;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.01049706950299176;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.04027073724177881;
                } else {
                  result[0] += 0.0978174483438034;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
                result[0] += -0.01373278585084206;
              } else {
                result[0] += -0.03701058378820878;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.11111388739435127;
      } else {
        result[0] += -0.010576057925360257;
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
      result[0] += -0.0009808938847549265;
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0028507060342000848;
            } else {
              result[0] += -0.007485560974486472;
            }
          } else {
            result[0] += 0.00410929046886094;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
              result[0] += 0.0043054170336979565;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.050556182861329013) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.03777022046337392;
                      } else {
                        result[0] += -0.09419803952587312;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8195080757141131) ) ) {
                        result[0] += 0.028822392216294526;
                      } else {
                        result[0] += -0.04391837925705173;
                      }
                    }
                  } else {
                    result[0] += -0.030890604320583055;
                  }
                } else {
                  result[0] += 0.14644769447692127;
                }
              } else {
                result[0] += -0.017727753630077593;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.04907476545463758;
                  } else {
                    result[0] += -0.010640183251963406;
                  }
                } else {
                  result[0] += 0.0037852974445324403;
                }
              } else {
                result[0] += 0.0056643992878042445;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.774546623229981357) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                    result[0] += -0.0070413193176408016;
                  } else {
                    result[0] += -0.09906381978857777;
                  }
                } else {
                  result[0] += 0.012097088390078017;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.367881059646607333) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.007652283884675929;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.02772311616653483;
                    } else {
                      result[0] += -0.024889935970491505;
                    }
                  }
                } else {
                  result[0] += 0.028636335667759097;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            result[0] += -0.027174069387633084;
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += 0.003452682766926995;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0008096299780899503;
                    } else {
                      result[0] += 0.06136608160559453;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.033886451343110885;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += -0.03432929789632373;
                      } else {
                        result[0] += 0.04253940489781432;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += -0.04638334908097083;
                      } else {
                        result[0] += 0.01362899529349097;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.012664173571636479;
              }
            } else {
              result[0] += -0.03459916664537666;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0123964224024751;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                result[0] += 0.04347978995816819;
              } else {
                result[0] += 4.198320306022861e-05;
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += -0.05568963126484863;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                result[0] += 0.005477799150408371;
              } else {
                result[0] += -0.027386877736033557;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            result[0] += -0.026102493011674895;
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.970085620880127397) ) ) {
                result[0] += -0.0848521584103118;
              } else {
                result[0] += -0.006218112417310166;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.03915909514919211;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
                  result[0] += 0.01339006239931419;
                } else {
                  result[0] += -0.006671187313955506;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.03702813363690743;
                  } else {
                    if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.015606786063028927;
                    } else {
                      result[0] += -0.0333995313024284;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.610145330429078037) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                      result[0] += 0.017823414935857203;
                    } else {
                      result[0] += 0.1793387746054038;
                    }
                  } else {
                    result[0] += -0.09600117284333408;
                  }
                }
              } else {
                result[0] += -0.0004211257916423887;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                  result[0] += -0.026376277499117296;
                } else {
                  result[0] += -0.0753554928818965;
                }
              } else {
                result[0] += 0.03372855364009839;
              }
            }
          } else {
            result[0] += -0.0026271663529751823;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
          result[0] += -0.00032337144047361595;
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05751732674239948;
                } else {
                  result[0] += -0.0041575683037693234;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03984229601120912;
                } else {
                  result[0] += -0.04132842411429351;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                result[0] += -0.029115501696844893;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.01723370126842184;
                } else {
                  result[0] += 0.016706395395967872;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
              result[0] += 0.04934344614778688;
            } else {
              result[0] += -0.0024284744723590003;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.90263271331787287) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
              result[0] += -0.009151574085796813;
            } else {
              result[0] += -0.04227671649307527;
            }
          } else {
            result[0] += 0.010446522534765275;
          }
        } else {
          result[0] += 0.002940786133733688;
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.004153468488638863;
              } else {
                result[0] += -0.042132584730037836;
              }
            } else {
              result[0] += -0.09284957835195068;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.187374591827394354) ) ) {
                result[0] += -0.007085952755134104;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.350240230560303178) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                        result[0] += 0.1467845566699229;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += -0.09173025857910058;
                        } else {
                          result[0] += 0.08209904037860066;
                        }
                      }
                    } else {
                      result[0] += -0.1284440398849212;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.375737190246582919) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                          result[0] += 0.002725847971108576;
                        } else {
                          result[0] += 0.14055525036853025;
                        }
                      } else {
                        result[0] += -0.058245941740707255;
                      }
                    } else {
                      result[0] += -0.06995026329821957;
                    }
                  }
                } else {
                  result[0] += 0.023288865088145812;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                  result[0] += 0.10478704123296548;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.219419956207276279) ) ) {
                      result[0] += 0.0063388152757423155;
                    } else {
                      result[0] += -0.15513420302756556;
                    }
                  } else {
                    result[0] += 0.14680049301379577;
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.04070433382365817;
                } else {
                  result[0] += 0.13867425346845702;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += -0.026127688611927536;
                      } else {
                        result[0] += -0.18978456949552647;
                      }
                    } else {
                      result[0] += 0.029658193952327584;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                      result[0] += -0.16807140175894186;
                    } else {
                      result[0] += -0.03476669567572709;
                    }
                  }
                } else {
                  result[0] += 0.030996131214070133;
                }
              } else {
                result[0] += 0.03239440265117496;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.002927002230594644;
              } else {
                result[0] += -0.036005757931637315;
              }
            }
          } else {
            result[0] += 0.015169636174397969;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.006969826739308383;
      } else {
        result[0] += -0.007065985410382009;
      }
    } else {
      result[0] += -0.00010726050883413344;
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0005339635322358467;
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        result[0] += 0.005466412524528904;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
              result[0] += 0.010165151785876557;
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.06299168328906751;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05653366218764869;
                } else {
                  result[0] += 0.03681954419348982;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.912734985351563388) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.014416082593981822;
                  } else {
                    result[0] += -0.021762918580961618;
                  }
                } else {
                  result[0] += 0.0013000613212575433;
                }
              } else {
                result[0] += -0.005859360104096653;
              }
            } else {
              result[0] += -0.023636617362018365;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                    result[0] += -0.011476315816190458;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.05689851871609994;
                    } else {
                      result[0] += -0.19048000615674662;
                    }
                  }
                } else {
                  result[0] += -0.11140599811087376;
                }
              } else {
                result[0] += -0.007312306407440507;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.230628252029419833) ) ) {
                      result[0] += 0.016669585698395186;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.986918687820435458) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.09133231264610503;
                        } else {
                          result[0] += -0.0025446350007993527;
                        }
                      } else {
                        result[0] += -0.17086269888192498;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.061799353701726095;
                    } else {
                      result[0] += 0.08615935975174203;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                    result[0] += 0.07274072330888186;
                  } else {
                    result[0] += -0.03011162755843303;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                  result[0] += -0.029569308586712303;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.051747083663941318) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.259080410003662998) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.158509254455567294) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                            result[0] += -0.005177947232431958;
                          } else {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += 0.08200919190178747;
                            } else {
                              result[0] += 0.012061154990497662;
                            }
                          }
                        } else {
                          result[0] += -0.07903228333835877;
                        }
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.14326446734056622;
                        } else {
                          result[0] += 0.01047310891991512;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.240144717803698;
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.901921629905701128) ) ) {
                          result[0] += 0.06077625102743981;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                            result[0] += 0.014588156434786993;
                          } else {
                            result[0] += -0.17366382691998367;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                        result[0] += -0.08613091970071207;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.008672604399926042;
                          } else {
                            result[0] += 0.11438523795621774;
                          }
                        } else {
                          result[0] += -0.012462341935844286;
                        }
                      }
                    } else {
                      result[0] += 0.029901359532372853;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.901921629905701128) ) ) {
                result[0] += -0.0016114266320083502;
              } else {
                result[0] += -0.008092965246393911;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.030873297076955908;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.07739773359484924;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                      result[0] += -0.017547889467331403;
                    } else {
                      result[0] += 0.0480174274335298;
                    }
                  }
                }
              } else {
                result[0] += 0.001331453251823979;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      result[0] += -0.0730520865831192;
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.947818994522095615) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.03168834871632659;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.23023242403430239;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                      result[0] += 0.013637462701100535;
                    } else {
                      result[0] += 0.13963892831061855;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.018533822780588474;
                  } else {
                    result[0] += 0.029869383183619508;
                  }
                }
              }
            } else {
              result[0] += -0.07950294671321162;
            }
          } else {
            result[0] += -0.03395421011429706;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.04034707904737374;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += 0.6963371501444273;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                    result[0] += -0.05722837427289155;
                  } else {
                    result[0] += 0.16522515788692194;
                  }
                } else {
                  result[0] += -0.008736227187314106;
                }
              }
            } else {
              result[0] += -0.04570660550031503;
            }
          }
        }
      } else {
        result[0] += -0.08089834580569072;
      }
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0005128128876576834;
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.003153410088903752;
              } else {
                result[0] += 0.12573596235017367;
              }
            } else {
              result[0] += -0.02098613341598833;
            }
          } else {
            result[0] += -0.07967438041832531;
          }
        } else {
          result[0] += 0.006170699662549327;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
            result[0] += 0.01658658581953899;
          } else {
            result[0] += 0.0012993488873130894;
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.309008359909058505) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += -0.021568010187230244;
                } else {
                  result[0] += 0.019877937150481126;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += 0.021484886501420025;
                      } else {
                        result[0] += -0.07436972562380652;
                      }
                    } else {
                      result[0] += 0.07087068026991904;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.05017307889606685;
                        } else {
                          result[0] += 0.08054872294208641;
                        }
                      } else {
                        result[0] += -0.010563219322570185;
                      }
                    } else {
                      result[0] += 0.003310693594877661;
                    }
                  }
                } else {
                  result[0] += 0.0007336681876749119;
                }
              }
            } else {
              result[0] += -0.03207332094025724;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
              result[0] += -0.0028477785652486283;
            } else {
              result[0] += 0.00833304719338321;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += 0.1435117020244632;
              } else {
                result[0] += 0.35115780258715673;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.08233642589828272;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.06867466804962978;
                } else {
                  result[0] += 0.08573133202496788;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
              result[0] += 0.0008934633828929106;
            } else {
              result[0] += 0.08623199873809584;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.04822896880666068;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                      result[0] += 0.010495967953391113;
                    } else {
                      result[0] += 0.062110883339153114;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += 0.1770440931283864;
                    } else {
                      result[0] += 0.07181426811834143;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.027471730688347136;
                  } else {
                    result[0] += 0.04907681503869325;
                  }
                }
              } else {
                result[0] += -0.00877863749128127;
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                result[0] += -0.012372794714552576;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.579273939132691318) ) ) {
                  result[0] += -0.05824739443612538;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += 0.025508756960515522;
                  } else {
                    result[0] += -0.038941447619428284;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.08809804358591825;
              } else {
                result[0] += -0.0033536261504211237;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0010368001212824006;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                result[0] += -0.01998072715686565;
              } else {
                result[0] += 0.037475094075101856;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    result[0] += 0.2533062715040806;
                  } else {
                    result[0] += 0.9391427854328263;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                        result[0] += 0.07087306281703985;
                      } else {
                        result[0] += 0.3143067810369911;
                      }
                    } else {
                      result[0] += 0.2818954343531441;
                    }
                  } else {
                    result[0] += 0.06453451282803545;
                  }
                }
              } else {
                result[0] += 0.0017017320093073197;
              }
            } else {
              result[0] += 0.03696874693657459;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                    result[0] += 0.02970263602900977;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.10717693113449961;
                    } else {
                      result[0] += 0.578355115849157;
                    }
                  }
                } else {
                  result[0] += 0.006508222837932825;
                }
              } else {
                result[0] += -0.05636562856218585;
              }
            } else {
              result[0] += -0.05179274080628245;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += -0.014631853637624684;
              } else {
                result[0] += -0.054118077188656556;
              }
            } else {
              result[0] += 0.07018359940917032;
            }
          }
        }
      }
    } else {
      result[0] += -0.053456751293449005;
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.0034374458008423385;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
              result[0] += -0.02770305756779448;
            } else {
              result[0] += 0.11239152036307581;
            }
          } else {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.036310994603189144;
            } else {
              result[0] += -0.0069074936556930144;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.743881702423096591) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.807895898818970615) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.011461368616943838;
              } else {
                result[0] += -0.007030736575137517;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.06037263260256365;
                  } else {
                    result[0] += -0.11526297933306434;
                  }
                } else {
                  result[0] += -0.0024295372143206613;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.783891201019287998) ) ) {
                  result[0] += 0.006353652307952623;
                } else {
                  result[0] += 0.13440436286991728;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.003350293178317878;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                    result[0] += -0.01770535484568001;
                  } else {
                    result[0] += -0.12328084851598481;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                    result[0] += 0.0491210347277945;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                        result[0] += 0.010932442530146283;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.0012996417057980715;
                        } else {
                          result[0] += -0.023564138901389125;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.06660373485446867;
                      } else {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.00752599500970662;
                        } else {
                          result[0] += -0.10192913827162958;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += -0.002374428307293548;
                      } else {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.0028962986520196772;
                        } else {
                          result[0] += -0.034031144037650785;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.017797946929933417) ) ) {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                                result[0] += 0.04227287370522598;
                              } else {
                                result[0] += -0.1154564103235971;
                              }
                            } else {
                              result[0] += 0.07610546465091349;
                            }
                          } else {
                            result[0] += -0.09467900042266225;
                          }
                        } else {
                          result[0] += -0.07687116151729217;
                        }
                      } else {
                        result[0] += 0.027342921090512354;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                      result[0] += 0.0076532122011300115;
                    } else {
                      result[0] += -0.0034328858276246204;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                    result[0] += 0.006208738424454562;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.0064570487485526296;
                    } else {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.013523119689170422;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                          result[0] += -0.041640964704773326;
                        } else {
                          result[0] += 0.06837288768086257;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                result[0] += -0.019736819419065927;
              } else {
                result[0] += 0.0004569769113464044;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.027170657809049904;
              } else {
                result[0] += -0.02507857071708205;
              }
            }
          } else {
            result[0] += -0.04975724050902625;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.03863632268163665;
          } else {
            result[0] += -0.007775796776124608;
          }
        } else {
          result[0] += -0.027156835846988338;
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.00624887735961921;
            } else {
              result[0] += -0.0454951406856409;
            }
          } else {
            result[0] += 0.015879021918510226;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.016407033057946135;
            } else {
              result[0] += -0.02431522967211814;
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.05435546271971431;
            } else {
              result[0] += 0.02088095723607749;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
          result[0] += 0.0034094603938378733;
        } else {
          result[0] += -0.003640662514504844;
        }
      } else {
        result[0] += -0.0018145197065352033;
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.00028720964909138174;
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.02446651458740412) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += 0.006072283578094812;
            } else {
              result[0] += 0.04617743640782286;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.665476083755494052) ) ) {
                result[0] += -0.0077401289352560425;
              } else {
                result[0] += -0.03293470681834506;
              }
            } else {
              result[0] += 0.01998346274429065;
            }
          }
        } else {
          result[0] += -0.018682865976257113;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
      result[0] += 0.00431014570612212;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.777674019336700661) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.030812958871032192;
        } else {
          result[0] += 0.07019534751550618;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.002519882644726496;
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.524296045303345615) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.298029184341431552) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
                    result[0] += 0.01877705183094788;
                  } else {
                    result[0] += -0.0977077109641003;
                  }
                } else {
                  result[0] += -0.13641362406566152;
                }
              } else {
                result[0] += 0.04794594737139603;
              }
            } else {
              result[0] += 0.06939750474926575;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.031158542582589984;
            } else {
              result[0] += 0.012271967664107522;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.0009955435572332162;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.004498415006196388;
              } else {
                result[0] += -0.017806325410609957;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.139882326126099521) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.11602659494805861;
          } else {
            result[0] += -0.02530147704909079;
          }
        } else {
          result[0] += 0.09875784071542382;
        }
      } else {
        result[0] += 0.07364692539511095;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.78399753570556818) ) ) {
            result[0] += 0.026100378124828405;
          } else {
            result[0] += 0.09754505475337732;
          }
        } else {
          result[0] += 0.003739969373052921;
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.0035774472885878187;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                    result[0] += 0.01921478223934427;
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.00988607779517369;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                        result[0] += 0.009827509502153617;
                      } else {
                        result[0] += -0.07987880783884818;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.03889746562997342;
                } else {
                  result[0] += -0.05426545729716675;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                result[0] += -0.003550157109326561;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.052547937162049564;
                } else {
                  result[0] += -0.016017307585756434;
                }
              }
            }
          } else {
            result[0] += -0.09182910977030223;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.7444796562194842) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.09443567192252963;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.509355545043946201) ) ) {
                    result[0] += 0.0036941083128575847;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.00544117000014059;
                    } else {
                      result[0] += 0.024171549521333133;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.0023875371646075597;
                } else {
                  result[0] += -0.03003603027254867;
                }
              }
            } else {
              result[0] += 0.07032849029713738;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.0018537809381936868;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.006168534613475835;
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.013169709708513867;
                    } else {
                      result[0] += -0.04856095914210609;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.048865041059417726;
                    } else {
                      result[0] += -0.0004545409582430017;
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.878078937530519354) ) ) {
                        result[0] += -0.04731961287958716;
                      } else {
                        result[0] += 0.05578332744474175;
                      }
                    } else {
                      result[0] += -0.059845372394956;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.417592287063599077) ) ) {
                  result[0] += 0.0013207953931454273;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.006392269384610884;
                        } else {
                          result[0] += 0.026890754909753924;
                        }
                      } else {
                        result[0] += 0.04380010341919106;
                      }
                    } else {
                      result[0] += -0.004531271360198127;
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                            result[0] += -0.04694784374966153;
                          } else {
                            result[0] += 0.015800431311353655;
                          }
                        } else {
                          result[0] += -0.06374383661271253;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.11177911413825713;
                            } else {
                              result[0] += -0.048872532301695026;
                            }
                          } else {
                            result[0] += 0.12424018729070563;
                          }
                        } else {
                          result[0] += 0.024397686475434714;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.0013153764663324497;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8195080757141131) ) ) {
                          result[0] += 0.012632248426687124;
                        } else {
                          result[0] += 0.055398556156524205;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.55753517150879084) ) ) {
              result[0] += -0.006088964042674702;
            } else {
              result[0] += 0.006696519770009988;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += 0.07381296085496161;
              } else {
                result[0] += -0.024072088191726633;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.002576828002931464) ) ) {
                result[0] += -0.07486898986829922;
              } else {
                result[0] += 0.05257922858344837;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.03916497618259524;
          } else {
            result[0] += 0.07237464525744987;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
          result[0] += -0.0017991706040028721;
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0011503143561076006;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05479049682617365) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11247110366821467) ) ) {
                  result[0] += -0.0027018614506610712;
                } else {
                  result[0] += 0.009958927004322847;
                }
              } else {
                result[0] += 0.03548622166969696;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += 0.010410322387497492;
              } else {
                result[0] += 0.051474365901024555;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
            result[0] += 0.0019112446537311936;
          } else {
            result[0] += 0.02979119653655359;
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.05976593279409194;
          } else {
            result[0] += 0.01935523687133706;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
          result[0] += -0.02510929621633754;
        } else {
          result[0] += 0.01179902826796249;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.497206687927246982) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.038412312270759856;
            } else {
              result[0] += 0.015458498423215075;
            }
          } else {
            result[0] += -0.0021792095258370684;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.006478893553982597;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += 0.026230109493203393;
              } else {
                result[0] += -0.01597878853119357;
              }
            }
          } else {
            result[0] += -0.018030674916921507;
          }
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.912734985351563388) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.11942149644483116;
                  } else {
                    result[0] += -0.005426292581229303;
                  }
                } else {
                  result[0] += -0.08391314611014833;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += 0.018673403826689687;
                } else {
                  result[0] += -0.05463165598039881;
                }
              }
            } else {
              result[0] += 0.004067067815576499;
            }
          } else {
            result[0] += 0.013250287162362316;
          }
        } else {
          result[0] += -0.0022283947302954373;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += 0.0007567484304209575;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.04126702413506822;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.003932043155508209;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.03138026055900551;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                    result[0] += 0.004746488315505285;
                  } else {
                    result[0] += 0.01825320122808974;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.014438432845418993;
              } else {
                result[0] += 0.0633926379911778;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.004351959222044495;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.036961450966168305;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.042097564684065156;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                          result[0] += -0.0561802955132401;
                        } else {
                          result[0] += -0.00909452074496439;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.004428472871682974;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.071567356586456743) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.08503081371264179;
                    } else {
                      result[0] += 0.0888811493721106;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.0011257542851069529;
                    } else {
                      result[0] += -0.018095268964750798;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.0785364889449619;
                    } else {
                      result[0] += -0.006055357137539241;
                    }
                  } else {
                    if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.036435339857313745;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.015701718986343254;
                      } else {
                        result[0] += -0.04235002810396726;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
            result[0] += -0.024621522496435136;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.01727535096585579;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.03600509764362344;
              } else {
                result[0] += -0.023933005803085966;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.97070193290710538) ) ) {
            result[0] += 0.005423183776578463;
          } else {
            result[0] += -0.017890489765195455;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0018449445863491865;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.391078948974610263) ) ) {
                  result[0] += 0.1042184129291946;
                } else {
                  result[0] += -0.051572984553750506;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                result[0] += -0.16786984973210364;
              } else {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.0004857148507151592;
                } else {
                  result[0] += 0.026582274773664538;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.132848501205445224) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                result[0] += -0.0006740910153801623;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
                  result[0] += -0.05087584924253539;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.016459637908620324;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                      result[0] += -0.017094337155454633;
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.03654120618673064;
                      } else {
                        result[0] += 0.01693904836583191;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.004714491700132708;
              } else {
                result[0] += 0.053561976375285286;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.0009031958354103642;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.15909472214767229;
                  } else {
                    result[0] += 0.024458714119300954;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.00805801676095397;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                        result[0] += -0.06766854361727005;
                      } else {
                        result[0] += 0.021532617053459116;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                          result[0] += -0.11688443499984146;
                        } else {
                          result[0] += 0.0032333869041027644;
                        }
                      } else {
                        result[0] += 0.015398392626224656;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.006785493910672917;
                    } else {
                      result[0] += 0.04322480013921979;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.08050654583448609;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.05807608591196831;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.6317791938781756) ) ) {
                      result[0] += -0.02686840985679927;
                    } else {
                      result[0] += 0.028418375504309215;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.026330214219241282;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.620046615600586826) ) ) {
                    result[0] += -0.008896107632202733;
                  } else {
                    result[0] += 0.02220467930192685;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                  result[0] += -0.05795049383117317;
                } else {
                  result[0] += 0.1325870316024855;
                }
              } else {
                result[0] += -0.007671702736511713;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
                result[0] += 0.008602116545358486;
              } else {
                result[0] += 0.10484517399423268;
              }
            }
          }
        } else {
          result[0] += -0.04320568423840868;
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.00984960571335234;
      } else {
        result[0] += -0.032227984183660206;
      }
    }
  } else {
    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.004771142970450298;
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0045603692493601305;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.01840511327816353;
                } else {
                  result[0] += 0.10510859151799255;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                  result[0] += 0.019170606532711378;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                    result[0] += -0.03654367339155167;
                  } else {
                    result[0] += 0.00027947246601271617;
                  }
                }
              }
            } else {
              result[0] += 0.021651941308453267;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.01275890007765494;
                } else {
                  result[0] += 0.002442545866540857;
                }
              } else {
                result[0] += -0.009815815650280689;
              }
            } else {
              result[0] += -0.02325449785241626;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += -0.002373699089064811;
            } else {
              result[0] += -0.032727660516727805;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.01507586779406222;
        } else {
          result[0] += -0.002988723733436785;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
          result[0] += 0.0014274487812543355;
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.009070314294968674;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.013152456905016095;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.015843741442450987;
              } else {
                result[0] += -0.0040879952306612415;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += 0.028320298024295577;
          } else {
            result[0] += -0.020101802990466898;
          }
        } else {
          result[0] += -0.0062727285772269286;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.825982809066773349) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.807895898818970615) ) ) {
              result[0] += -0.004122104989921333;
            } else {
              result[0] += 0.0027484376059360373;
            }
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.016780374058912344;
            } else {
              result[0] += 0.02259912691925825;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04508973172415776;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.014292918603417094;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += -0.08153214411285317;
                    } else {
                      result[0] += -0.0028829228722114953;
                    }
                  }
                } else {
                  result[0] += 0.017306359112151667;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.0014014431072986444;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.028270574602375316;
                  } else {
                    result[0] += -0.004894326518720727;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.0014790348085087295;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
                      result[0] += -0.1054571277943121;
                    } else {
                      result[0] += -0.001970325427380358;
                    }
                  } else {
                    result[0] += -0.04486164087787245;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.15096005987339856;
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.83939445018768355) ) ) {
                            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.10517444885957908;
                            } else {
                              result[0] += -0.0016875590259206102;
                            }
                          } else {
                            result[0] += 0.008903862217562867;
                          }
                        }
                      } else {
                        result[0] += 0.0007983628578263121;
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.015087146347121947;
                      } else {
                        result[0] += 0.0369062883995752;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += 0.013364358418858777;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.11326837539672896) ) ) {
                          result[0] += -0.026439789926036295;
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                            result[0] += -0.018623013221650397;
                          } else {
                            result[0] += 0.012905562898626599;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                          result[0] += -0.02289049565382993;
                        } else {
                          result[0] += 0.017711664620747657;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08890932561925131;
                  } else {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.009432241214878892;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.01244453318472258;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                          result[0] += 0.026846509003500964;
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.007639297596448997;
                          } else {
                            result[0] += 0.08723191965301616;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.007348568537644907;
              } else {
                result[0] += 0.03121290383394945;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.03450528534554122;
          } else {
            result[0] += -0.004391986078187848;
          }
        } else {
          result[0] += -0.02313264536836035;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.10975960052748109;
              } else {
                result[0] += -0.04121224672514211;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.10071849371113402;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.04552083368418513;
                  } else {
                    result[0] += 0.08414454516997508;
                  }
                }
              } else {
                result[0] += 0.016058662202525296;
              }
            }
          } else {
            result[0] += 0.008360844644427254;
          }
        } else {
          result[0] += 0.058930848133657836;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      result[0] += -0.00020716208699425817;
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.00020423247331329016;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.02446651458740412) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
              result[0] += -0.00012834609013861328;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += -0.01778401975294447;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.007280497446668988;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.051945771061634644;
                  } else {
                    if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.010724266216717711;
                        } else {
                          result[0] += 0.07092384874969994;
                        }
                      } else {
                        result[0] += 0.1054895098636038;
                      }
                    } else {
                      result[0] += 0.006920488241602104;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.021832109891533835;
          }
        } else {
          result[0] += 0.011810674143583465;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0010644789255837453;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.07467871130695761;
              } else {
                result[0] += 0.09132345921682428;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.13960093388368036;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.10645562637251048;
                } else {
                  result[0] += -0.02511206228110052;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += 0.11551969503903062;
              } else {
                result[0] += 0.005723588990585671;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.084203958511353427) ) ) {
                result[0] += -0.005627997724983326;
              } else {
                result[0] += 0.0023520387339419427;
              }
            }
          } else {
            result[0] += 0.002277746065070731;
          }
        }
      } else {
        result[0] += -0.005447392417003449;
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.008647711548397167;
      } else {
        result[0] += 0.05583593824529233;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.0006719467844354185;
      } else {
        result[0] += -0.015499100372341007;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.02612728307088415;
                    } else {
                      result[0] += -0.005936715375424668;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.695412874221802646) ) ) {
                        result[0] += 0.00669387459994239;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.1362050877236882;
                        } else {
                          result[0] += -0.005065972137598498;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.10304688677362915;
                        } else {
                          result[0] += -0.02095894296098578;
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.005624211251049567;
                        } else {
                          result[0] += -0.048311640583410906;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.04232555858402974;
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.09522038652916497;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.01944907807407145;
                  } else {
                    result[0] += -0.05968405004029054;
                  }
                }
              }
            } else {
              result[0] += 0.0038353441255190787;
            }
          } else {
            result[0] += 0.0825787496327359;
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.441728115081787998) ) ) {
              result[0] += -0.0052598845340681745;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += -0.019457728720377217;
                } else {
                  result[0] += 0.1269468174006457;
                }
              } else {
                result[0] += -0.021517781368639674;
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.039087707788893186;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += 0.01760208634747496;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.04961470544125349;
                      } else {
                        result[0] += -0.008110677335816115;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                        result[0] += -0.02010842535336957;
                      } else {
                        result[0] += 0.017837220902103112;
                      }
                    }
                  } else {
                    result[0] += -0.046339385629523604;
                  }
                }
              }
            } else {
              result[0] += 0.018905752736199974;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                  result[0] += -0.1677597052503849;
                } else {
                  result[0] += -0.0007485380930188155;
                }
              } else {
                result[0] += 0.02498157646825205;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.010164116568231706;
              } else {
                result[0] += 0.00993221644389935;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.008565711830068528;
            } else {
              result[0] += -0.0551996461870571;
            }
          }
        } else {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.004807788304224583;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.02985057398050997;
                } else {
                  result[0] += 0.027976609885005378;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += -0.053825378254878015;
                } else {
                  result[0] += -0.0220761220803372;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.101640701293947089) ) ) {
                      result[0] += -0.02935200974515767;
                    } else {
                      result[0] += 0.001243803032235228;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += 0.09855949985974106;
                    } else {
                      result[0] += 0.017674982972522943;
                    }
                  }
                } else {
                  result[0] += -0.010989283117670726;
                }
              } else {
                result[0] += -0.0394324442328135;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += 0.1266335795000468;
              } else {
                result[0] += 0.020389159310260748;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
    result[0] += 0.0002185050462274304;
  } else {
    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10223627090454279) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += 0.01951044087577554;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.120439291000367099) ) ) {
              result[0] += 0.0039059360529633946;
            } else {
              result[0] += -0.019112775704205245;
            }
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
            result[0] += -0.09974597140203918;
          } else {
            result[0] += -0.029586154331980127;
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.021639209834640207;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.03539302610758218;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
                    result[0] += 0.012160056366560758;
                  } else {
                    result[0] += -0.0002489004572989141;
                  }
                }
              } else {
                result[0] += -0.05892134703284786;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.628555774688722479) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.05660856689722781;
              } else {
                result[0] += -0.005608747023254403;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                result[0] += -0.008487346004050189;
              } else {
                result[0] += -0.05922177922367244;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.07035839755129601;
            } else {
              if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
                    result[0] += -0.056698374498563076;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                      result[0] += -0.04342366459232916;
                    } else {
                      result[0] += 0.05698568880350577;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.03886513576793864;
                  } else {
                    result[0] += -0.08255666413288229;
                  }
                }
              } else {
                result[0] += 0.16927243840165684;
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.179772853851319248) ) ) {
                  result[0] += 0.0010467841778287405;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                    result[0] += -0.024894599362179443;
                  } else {
                    result[0] += 0.054858420811612;
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                    result[0] += 0.005545324431560114;
                  } else {
                    result[0] += 0.04777979309191155;
                  }
                } else {
                  result[0] += 0.10177969100250446;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
                  result[0] += -0.031800799117092084;
                } else {
                  result[0] += -0.010504580339295412;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.03777202294941062;
                } else {
                  result[0] += 0.015336542628284353;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
          result[0] += -0.07113281527540948;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            result[0] += -0.06495311201517644;
          } else {
            result[0] += 0.02886413563146691;
          }
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
              result[0] += 0.2186203279407376;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.860674262046814409) ) ) {
                result[0] += -0.03523307898085962;
              } else {
                result[0] += -0.012402798385340743;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.00037091725247653833;
              } else {
                result[0] += -0.09081386151371683;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.03696074294693798;
              } else {
                result[0] += 0.10454647655688497;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.008985634905008017;
                  } else {
                    result[0] += 0.05861232118902581;
                  }
                } else {
                  result[0] += -0.06802126858273158;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.0020621404534352417;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.847591876983644354) ) ) {
                      result[0] += -0.0992582513320969;
                    } else {
                      result[0] += -0.005211347530922164;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.221818685531617099) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.867973804473877841) ) ) {
                        result[0] += 0.0016749330351668642;
                      } else {
                        result[0] += 0.10615765313834978;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.825982809066773349) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                          result[0] += -0.0064698652080090084;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.641049861907960761) ) ) {
                            result[0] += 0.05835458384708182;
                          } else {
                            result[0] += -0.03198883331239568;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11607551574707209) ) ) {
                          result[0] += -0.08005529728036083;
                        } else {
                          result[0] += -0.022490515023700662;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.008345623729003304;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
                        result[0] += -0.12278985689842344;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.736968994140626776) ) ) {
                          result[0] += -0.10268636072520546;
                        } else {
                          result[0] += 0.05573853184479112;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.05403729141552103;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
              result[0] += -0.03393291482965363;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += 0.12980630646073996;
              } else {
                result[0] += 0.03162359071518072;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
      result[0] += 0.00011731480429053381;
    } else {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.089659929275513583) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.025569091329077767;
              } else {
                result[0] += 0.11699267195077985;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.03339382042262925;
                } else {
                  result[0] += -0.03915323268640622;
                }
              } else {
                result[0] += -0.0007247796977515494;
              }
            }
          } else {
            result[0] += 0.0681742284254801;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
            result[0] += -0.03939878563013408;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.023474786742369403;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.015730651431873088;
              } else {
                result[0] += 0.021828662405501534;
              }
            }
          }
        }
      } else {
        result[0] += -0.00867367700856752;
      }
    }
  } else {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
              result[0] += 0.006787182180644901;
            } else {
              result[0] += -0.06198879192032212;
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.06215608747949876;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.01532917965474738;
                    } else {
                      result[0] += 0.0007098916273405861;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.004834802281870486;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.018615982964991253;
                      } else {
                        result[0] += 0.023260875817057425;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                          result[0] += -0.02200757455115336;
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.044933628013104604;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += 0.04960687615078483;
                            } else {
                              result[0] += 0.02634322476667685;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                            result[0] += -0.12046606924413317;
                          } else {
                            result[0] += -0.003118772934122062;
                          }
                        } else {
                          result[0] += 0.12663952234494572;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.052626306889007636;
                } else {
                  result[0] += -0.06106205499832939;
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
                      result[0] += 0.013341095292374566;
                    } else {
                      result[0] += -0.00837049728157743;
                    }
                  } else {
                    result[0] += -0.012231326634051595;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                      result[0] += 0.003835233726187152;
                    } else {
                      result[0] += -0.03357706531350623;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                      result[0] += 0.030368342245295166;
                    } else {
                      result[0] += 0.005608945066480655;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.030589674243864873;
                    } else {
                      result[0] += -0.06035240590398203;
                    }
                  } else {
                    result[0] += -0.01166538313851204;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                        result[0] += -0.015525961705040836;
                      } else {
                        result[0] += 0.032928642191944406;
                      }
                    } else {
                      result[0] += 0.873163375068153;
                    }
                  } else {
                    result[0] += -0.010151087072391372;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.08134942008036787;
                } else {
                  result[0] += 0.0011302704161612104;
                }
              } else {
                result[0] += -0.0029044355970493704;
              }
            } else {
              result[0] += 0.018131516249921553;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
              result[0] += 0.013298831454446325;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += 0.045647622232309774;
              } else {
                result[0] += -0.06064887386537087;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.006194179626268566;
        } else {
          result[0] += -0.036237652089358335;
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
        result[0] += -0.02575727160057728;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                result[0] += 0.0074026414941332636;
              } else {
                result[0] += 0.20111927445047023;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += 0.12106551088489184;
              } else {
                result[0] += 0.03131311808787588;
              }
            }
          } else {
            result[0] += -0.007923203428191306;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              result[0] += -0.016847645181237457;
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.05184453705315495;
              } else {
                result[0] += 0.03980441419026519;
              }
            }
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.006228488815332619;
            } else {
              result[0] += 0.12081371277570507;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.023256675454796275;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.09519884179810056;
                } else {
                  result[0] += 0.04747710126047583;
                }
              }
            } else {
              result[0] += 0.01537883762671507;
            }
          } else {
            result[0] += 0.004329867270930529;
          }
        } else {
          result[0] += 0.00028037323526155497;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.978102684020996982) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.417592287063599077) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                    result[0] += 0.014597814535794666;
                  } else {
                    result[0] += 0.04072739578917489;
                  }
                } else {
                  result[0] += 0.11634723516027781;
                }
              } else {
                result[0] += 0.0015408449502157624;
              }
            } else {
              result[0] += -0.0004342934997968657;
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                result[0] += 0.0017369802306287899;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.030109669362982434;
                } else {
                  result[0] += 0.006231009179522105;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                result[0] += 0.07054943130932016;
              } else {
                result[0] += 0.0069958810496937324;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.007648227758706855;
            } else {
              result[0] += -0.048775580146762226;
            }
          } else {
            result[0] += -0.014547673407609388;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
              result[0] += 0.06906012111460237;
            } else {
              result[0] += -0.089986613479789;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.028458703612629255;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.12972677627654747;
              } else {
                result[0] += 0.007028548656115662;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.008049070757684107;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.13594869126124173;
            } else {
              result[0] += -0.020262131280807227;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.060065006856259484;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                result[0] += -0.002173960423815033;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                    result[0] += -0.03978791576216539;
                  } else {
                    result[0] += -0.008865743098588213;
                  }
                } else {
                  result[0] += 0.17861465512489264;
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.013535567667709886;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.17710595223627348;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.749261140823365146) ) ) {
                        result[0] += 0.009559026510194572;
                      } else {
                        result[0] += 0.11132906375076333;
                      }
                    }
                  } else {
                    result[0] += -0.07078246451055321;
                  }
                }
              } else {
                result[0] += -0.04548617387631654;
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.10420077181417778;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += -0.01415590916938461;
                } else {
                  result[0] += -0.06235324220837771;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.012750233032888234;
              } else {
                result[0] += -0.03999008655373043;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.551017761230469638) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.11198686778003375;
        } else {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.665476083755494052) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.1027512952340864;
                  } else {
                    result[0] += -0.005350656835596555;
                  }
                } else {
                  result[0] += 0.01780938128778845;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.892608642578125888) ) ) {
                  result[0] += -0.16699091530715757;
                } else {
                  result[0] += -0.04935191967320267;
                }
              }
            } else {
              result[0] += 0.13012462793434912;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.11257260050116308;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                  result[0] += 0.10729252040547287;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.431880712509156162) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += 0.15513605742054748;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.076374173686519;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11815500259399592) ) ) {
                          result[0] += 0.03267836210860619;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.12582111358642756) ) ) {
                              result[0] += 0.08346869878939178;
                            } else {
                              result[0] += -0.067780863101441;
                            }
                          } else {
                            result[0] += -0.039384074108007844;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.12098776158034862;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                  result[0] += -0.1331772544842138;
                } else {
                  result[0] += 0.02247718175296111;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.051898827928746866;
      }
    } else {
      result[0] += 0.0003637366880496433;
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
      result[0] += 0.0003432126210214843;
    } else {
      result[0] += 0.009802087928273088;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.01036307819844285;
            } else {
              result[0] += 0.0509663859485317;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.009673264217037993;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += 0.10791681056993058;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.029425964888534457;
                } else {
                  result[0] += -0.006019976859591162;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.04122696071521491;
              } else {
                result[0] += -0.0011755707874833272;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.18957764445045788;
                } else {
                  result[0] += 0.004898769805007153;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.048477266506936734;
                } else {
                  result[0] += 0.03383169602497043;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.046281593509238016;
                } else {
                  result[0] += -0.10362332170879299;
                }
              } else {
                result[0] += 0.012710903359635256;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                result[0] += -0.03980121337678112;
              } else {
                result[0] += -0.0020280489285079317;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.030795218121091657;
              } else {
                result[0] += -0.00037415971090504994;
              }
            } else {
              result[0] += 0.00692252669966654;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.407877445220948154) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                      result[0] += 0.010487370540501285;
                    } else {
                      result[0] += -0.009412551657530136;
                    }
                  } else {
                    result[0] += 0.017561831556991647;
                  }
                } else {
                  result[0] += 0.02138794055034407;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.03247555475807585;
                } else {
                  result[0] += -0.01642289884204053;
                }
              }
            } else {
              result[0] += -0.02043447253043463;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                result[0] += -0.011961794765489264;
              } else {
                result[0] += 0.05405266219139365;
              }
            } else {
              result[0] += 0.07953676316719016;
            }
          } else {
            result[0] += -0.019841375468921438;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.07726197528638241;
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0011221433749536579;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                  result[0] += 0.0201306628203075;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.01849986647487514;
                  } else {
                    result[0] += -0.039418693725298465;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0015851134046564264;
                  } else {
                    result[0] += -0.050300235163230414;
                  }
                } else {
                  result[0] += 0.06894241640094516;
                }
              } else {
                result[0] += -0.0008482999234410552;
              }
            }
          } else {
            result[0] += -0.03071193549971036;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += -0.02193485128964619;
              } else {
                result[0] += 0.0005600780811758089;
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                  result[0] += -0.07225689088717223;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.044737342932330566;
                  } else {
                    result[0] += 0.09851104203341043;
                  }
                }
              } else {
                result[0] += -0.008942549514462054;
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.011300526296134528;
            } else {
              result[0] += -0.03725053856201145;
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.04582164212873381;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                    result[0] += 0.055491624979971045;
                  } else {
                    result[0] += 0.015442834572584037;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.313699722290040839) ) ) {
                  result[0] += -0.004458732997560854;
                } else {
                  result[0] += -0.0357136296833065;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += -0.0036323402309366573;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.034809792051672395;
                  } else {
                    result[0] += 0.012225694820100809;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.184114694595337802) ) ) {
                    result[0] += 0.029348389947416887;
                  } else {
                    result[0] += 0.09283043911311674;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.0019318704953292705;
            } else {
              result[0] += -0.019572038563942965;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += -0.0053857995362621;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.03343397382943395;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.014774248010846909;
                  } else {
                    result[0] += -0.049261872934252667;
                  }
                } else {
                  result[0] += -0.001548048977421406;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                result[0] += 0.0003296216109644136;
              } else {
                result[0] += -0.055983611701058436;
              }
            }
          }
        } else {
          result[0] += 0.0016746241650278841;
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
            result[0] += -0.053078632504923784;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                result[0] += 0.03709571225069568;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                      result[0] += 0.014070869277926097;
                    } else {
                      result[0] += -0.006003323366002978;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                      result[0] += 0.06356125996310857;
                    } else {
                      result[0] += -0.022942104335630747;
                    }
                  }
                } else {
                  result[0] += -0.03625704917344482;
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.019425129530295893;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.83939445018768355) ) ) {
                  result[0] += -0.05795210444951337;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                    result[0] += -0.03481017725950716;
                  } else {
                    result[0] += 0.008748534765051377;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.011555812743462691;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                    result[0] += -0.08319781851860786;
                  } else {
                    result[0] += 0.024542984384972672;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                          result[0] += -0.12084426352775063;
                        } else {
                          if ( UNLIKELY(  (data[38].missing != -1) && (data[38].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += 0.06828419802083205;
                          } else {
                            result[0] += -0.0666890634763222;
                          }
                        }
                      } else {
                        result[0] += 0.0027670395052247403;
                      }
                    } else {
                      result[0] += 0.02482636425390368;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                          result[0] += -0.06837744956999647;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                            result[0] += 0.1062263623248781;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                              result[0] += -0.08504398446072466;
                            } else {
                              result[0] += -0.014029116206408343;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.0003623682241185038;
                      }
                    } else {
                      result[0] += 0.05237065988282742;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.01333665958710921;
                } else {
                  result[0] += -0.03244578439569395;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += 0.022617753570452093;
                  } else {
                    result[0] += 0.15178922425282437;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8195080757141131) ) ) {
                    result[0] += 0.012255367826513135;
                  } else {
                    result[0] += -0.025122357499306677;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.0318107841298282;
                  } else {
                    result[0] += -0.04604763477938191;
                  }
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.04917398187689448;
                  } else {
                    result[0] += -0.008740077387589466;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
                result[0] += -0.03499419686651958;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.842459201812745917) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.11529975895110876;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                        result[0] += -0.053986125085674436;
                      } else {
                        result[0] += 0.01004164413365482;
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.07989690882069445;
                      } else {
                        result[0] += -0.03436380088618238;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.11074769842794228;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += -0.06918405269563185;
                        } else {
                          result[0] += 0.04650598333020986;
                        }
                      } else {
                        result[0] += 0.07917016794047958;
                      }
                    } else {
                      result[0] += 0.019605491520060823;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -8.240662917673866e-05;
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.019688847836200583;
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.006118811173852274;
          } else {
            result[0] += -0.02505582070803485;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.0003402285127465921;
        } else {
          result[0] += 0.0250196573397881;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
        result[0] += -0.2712174643413637;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
          result[0] += 0.07039436020852664;
        } else {
          result[0] += -0.027264816894655028;
        }
      }
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.780892848968506748) ) ) {
        result[0] += -2.6829206346821208e-05;
      } else {
        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
              result[0] += -0.023921971608212873;
            } else {
              result[0] += -0.008047530726720686;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.007144265048541387;
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.05285299646658991;
                        } else {
                          result[0] += -0.16787414757011376;
                        }
                      } else {
                        result[0] += -0.007468226045259677;
                      }
                    }
                  } else {
                    result[0] += 0.007383243303982054;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.055073267942418486;
                  } else {
                    result[0] += -0.01825133123006628;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                  result[0] += 0.0001365258757260348;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.02727386586724892;
                  } else {
                    result[0] += 0.14640461359645593;
                  }
                }
              }
            } else {
              result[0] += 0.011979856188572358;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.05594176118205038;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
              result[0] += 0.0015688182749135464;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.008247564711152847;
              } else {
                result[0] += 0.04828805324016299;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.0184845111758712;
        } else {
          result[0] += 0.005267705239860874;
        }
      } else {
        result[0] += 0.00034251357698138413;
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      result[0] += -0.04858660293834996;
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.05367376872612539;
            } else {
              result[0] += 0.08362688243359906;
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.09453923778096258;
            } else {
              result[0] += 0.03558473588296919;
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                  result[0] += 0.04043932861531356;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.737603187561036044) ) ) {
                    result[0] += -0.0642783266650731;
                  } else {
                    result[0] += 0.04687810921042468;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801954269409180576) ) ) {
                      result[0] += -0.017025186126716788;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.591613531112671787) ) ) {
                        result[0] += 0.05050448272443739;
                      } else {
                        result[0] += -0.045913573140988456;
                      }
                    }
                  } else {
                    result[0] += -0.039878017443910535;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.02174656499756392;
                  } else {
                    result[0] += -0.058028354319584655;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.69067406654357999) ) ) {
                  result[0] += -0.01728598742756868;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                          result[0] += 0.14731726381315402;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                            result[0] += -0.04200183826629754;
                          } else {
                            result[0] += 0.03515146204486088;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
                            result[0] += 0.011246953147598654;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.014138812136038528;
                            } else {
                              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                                  result[0] += -0.05846706798582742;
                                } else {
                                  result[0] += 0.14155387142502027;
                                }
                              } else {
                                result[0] += 0.2130898430892963;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                            result[0] += -0.07318218573172112;
                          } else {
                            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                                result[0] += 0.025753418261129236;
                              } else {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                    result[0] += -0.018041215395404403;
                                  } else {
                                    result[0] += 0.031209182643201785;
                                  }
                                } else {
                                  result[0] += -0.04174122176926179;
                                }
                              }
                            } else {
                              result[0] += 0.4168239223659693;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += 0.041062790539395655;
                    }
                  } else {
                    result[0] += -0.04503087855718729;
                  }
                }
              } else {
                result[0] += -0.04676619908940579;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += 0.011592327191288743;
            } else {
              result[0] += 0.0953992360197606;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.009949382533621171;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
                result[0] += 0.07488615196805581;
              } else {
                result[0] += 0.3094216372857303;
              }
            } else {
              result[0] += -0.03365601481771265;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.11990808661209196;
              } else {
                result[0] += -0.018448620871606182;
              }
            } else {
              result[0] += -0.06224248951098829;
            }
          } else {
            result[0] += -0.06358249185375382;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.00041811324474044814;
  } else {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.006168079324381381;
        } else {
          result[0] += -0.03315541851600721;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.080866575241089755) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.02676544243378892;
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.018795234969476472;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                              result[0] += 0.00992819107745801;
                            } else {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += -0.08189830528485204;
                              } else {
                                result[0] += 0.0353557659347672;
                              }
                            }
                          } else {
                            result[0] += 0.04864178559337263;
                          }
                        }
                      } else {
                        result[0] += 0.016077612389148465;
                      }
                    } else {
                      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.0050393431248688895;
                      } else {
                        result[0] += 0.014307856160090136;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.93885374069213956) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0021470944625328146;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += 0.01699501507488187;
                      } else {
                        result[0] += 0.06372830363482522;
                      }
                    }
                  } else {
                    result[0] += -0.02829486105435989;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0015711144177885253;
                  } else {
                    result[0] += -0.03288665869383204;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                    result[0] += 0.019760576010290732;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                      result[0] += 0.0005369479419642309;
                    } else {
                      result[0] += -0.06272678930973434;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.02852917127714257;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += -0.012695569603953275;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                      result[0] += 0.15272698436540474;
                    } else {
                      result[0] += 0.022892975275620453;
                    }
                  } else {
                    result[0] += -0.005597769765521411;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.007819725232902894;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04325453584846876;
                    } else {
                      result[0] += 0.012664704647761274;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.009078552912657831;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.051510764735472614;
                        } else {
                          result[0] += 0.01650696348537556;
                        }
                      } else {
                        result[0] += 0.03195023305082745;
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.006412148922864579;
                      } else {
                        result[0] += 0.01687935903783198;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.0631013787359135;
                  } else {
                    result[0] += -0.08254587466877578;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                      result[0] += -0.029087434440747042;
                    } else {
                      result[0] += 0.004753541980951875;
                    }
                  } else {
                    result[0] += -0.020120285706896477;
                  }
                } else {
                  result[0] += -0.036883125939012384;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += 0.057026172706167694;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                    result[0] += -0.08455086933134556;
                  } else {
                    result[0] += 0.003142432314294719;
                  }
                } else {
                  result[0] += -0.013322014815465986;
                }
              }
            } else {
              result[0] += 0.022810360051684773;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.025735538718695092;
                  } else {
                    result[0] += 0.0716465137304175;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.027437530655863754;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += -0.1632889696466176;
                    } else {
                      result[0] += 0.05647206196735108;
                    }
                  }
                }
              } else {
                result[0] += -0.008792922749248686;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0008552550728824429;
              } else {
                result[0] += 0.06840285778614104;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.030680605049038606;
          } else {
            result[0] += -0.0011280526449983655;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            result[0] += -0.02436844771290753;
          } else {
            result[0] += 0.01117338795733289;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
          result[0] += -0.026457069654699968;
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.01274978876046837;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0424223266282956;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.06601384357848517;
              } else {
                result[0] += -0.010493089079772868;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
      result[0] += 0.00026472493967844447;
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.591613531112671787) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.724856853485109198) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.916317939758301669) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.004492156099894559;
                } else {
                  result[0] += -0.022271567864662;
                }
              } else {
                result[0] += -0.036593822716595205;
              }
            } else {
              result[0] += -0.04987488921452961;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.03962992257535502;
              } else {
                result[0] += -0.0610727634130048;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.725620865821838823) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.941534638404846635) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                      result[0] += -0.016297831162595155;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                        result[0] += -0.08479285946056495;
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                          result[0] += -0.07709382844284274;
                        } else {
                          result[0] += 0.024086319497191533;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.0030305183278151408;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                    result[0] += 0.02378947166057431;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        result[0] += -0.4304767512080487;
                      } else {
                        result[0] += -0.09352949255557585;
                      }
                    } else {
                      result[0] += 0.08058340438025029;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.01634240150451749) ) ) {
                        result[0] += 0.005022969807153889;
                      } else {
                        result[0] += -0.1356001536963722;
                      }
                    } else {
                      result[0] += 0.055870299541379745;
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.18951576729388747;
                    } else {
                      result[0] += -0.02082826499274375;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                      result[0] += 0.03335446368318545;
                    } else {
                      result[0] += -0.08254873670811534;
                    }
                  } else {
                    result[0] += 0.07226216547482409;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
            result[0] += -0.027454204438887506;
          } else {
            result[0] += -0.0939959705642176;
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.381086945533752885) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04949659891752281;
            } else {
              result[0] += -0.007864813944664562;
            }
          } else {
            result[0] += 0.12826284710865765;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.07512241481727579;
            } else {
              result[0] += 0.11436723874877533;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.325443029403687412) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.059420347213746005) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.388278961181641513) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          result[0] += 0.09629693932496314;
                        } else {
                          result[0] += 0.00020568095953410337;
                        }
                      } else {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.025619547095563536;
                            } else {
                              result[0] += 0.11985533178786921;
                            }
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
                              result[0] += -0.019848963610960304;
                            } else {
                              result[0] += 0.20068539771141475;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                              result[0] += -0.0480421305659733;
                            } else {
                              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                                result[0] += 0.11639080123654322;
                              } else {
                                result[0] += 0.013672707579917072;
                              }
                            }
                          } else {
                            result[0] += -0.04775943923650291;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.10759154389571257;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                      result[0] += 0.06834578807879162;
                    } else {
                      result[0] += -0.023638213798592968;
                    }
                  }
                } else {
                  result[0] += 0.07633030360579325;
                }
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.07294151659408435;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.010810243320182891;
                    } else {
                      result[0] += -0.07415588490647444;
                    }
                  } else {
                    result[0] += 0.04286013461309712;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.08036335307327026;
                } else {
                  result[0] += -0.013558724865794817;
                }
              } else {
                result[0] += -0.1299810879256478;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
          result[0] += -0.026753985831753896;
        } else {
          result[0] += 0.027530460416564824;
        }
      } else {
        result[0] += -0.05995204454649004;
      }
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.297764539718628818) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.06644128168866903;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += 0.04973122272810221;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.006223597451962795;
              } else {
                result[0] += 0.05924453792510695;
              }
            }
          }
        } else {
          result[0] += -0.037159034201246804;
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.05423478141665275;
        } else {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.08175068175177798;
          } else {
            result[0] += -0.032596688836579055;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.298283100128175604) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.01049994779037899;
          } else {
            result[0] += -0.03981457006329616;
          }
        } else {
          result[0] += -0.00967757676581617;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.561121463775635654) ) ) {
              result[0] += -0.17058942929928878;
            } else {
              result[0] += -0.007973486701729128;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.07858228683471857) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                  result[0] += 0.1071727426540057;
                } else {
                  result[0] += -0.07836876645622415;
                }
              } else {
                result[0] += 0.065403374182935;
              }
            } else {
              result[0] += -0.14895708951336126;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.310138225555421698) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.0074795560424222515;
                } else {
                  result[0] += -0.12895034580055062;
                }
              } else {
                result[0] += 0.010461869142445047;
              }
            } else {
              result[0] += 0.09509577573108646;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.24492526054382413) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                        result[0] += 0.053387085474566935;
                      } else {
                        result[0] += 0.009884208053411244;
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.017784042170176137;
                      } else {
                        result[0] += -0.007398895208518625;
                      }
                    }
                  } else {
                    result[0] += -0.00013909455333587523;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += 0.07639699760623281;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.06795853896832743;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                          result[0] += -0.057930205217119723;
                        } else {
                          result[0] += 0.03189850918685148;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
                      result[0] += 0.026237352450487292;
                    } else {
                      result[0] += -0.011079041159586321;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.092111126594706;
                    } else {
                      result[0] += -0.0012331574425160328;
                    }
                  } else {
                    result[0] += 0.0037550051565316067;
                  }
                } else {
                  result[0] += -0.01970789987282693;
                }
              }
            } else {
              result[0] += -0.022511585816994;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.004944801316050903;
          } else {
            result[0] += -0.05579401738822383;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += -0.06818413904811636;
          } else {
            result[0] += -0.0064674567074279295;
          }
        }
      } else {
        result[0] += -0.012637207692029542;
      }
    }
  } else {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
        result[0] += -0.013404716975356777;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.184114694595337802) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
            result[0] += 0.009884045560183964;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                  result[0] += 0.12600618897823188;
                } else {
                  result[0] += 0.023471808381485177;
                }
              } else {
                result[0] += -0.13441257516435431;
              }
            } else {
              result[0] += 0.08691102518180888;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += -0.16083177795452583;
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.01453650972278852;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.774546623229981357) ) ) {
                result[0] += -0.07845466406226494;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                  result[0] += -0.011000244120887749;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.003721241095780632;
                  } else {
                    result[0] += 0.0789628508606884;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
              result[0] += 0.007963519567194101;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.06983659630109704;
                } else {
                  result[0] += -0.0013411494588429027;
                }
              } else {
                result[0] += -0.024798640411327164;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
              result[0] += -0.002695978818974093;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.00807092736937493;
              } else {
                result[0] += -0.0380261811393767;
              }
            }
          }
        } else {
          result[0] += -0.06286610334525602;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.289602279663086826) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.149111986160279208) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                  result[0] += 0.015627177282479938;
                } else {
                  result[0] += -0.05389187374539487;
                }
              } else {
                result[0] += -0.004261543581430951;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.02430779860740659;
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.008158950291840805;
                } else {
                  result[0] += 0.005895886618434425;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.509355545043946201) ) ) {
              result[0] += 0.002527600643840319;
            } else {
              result[0] += 0.02121894146326337;
            }
          }
        } else {
          result[0] += 6.730489166183966e-05;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.140846252441408026) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.158509254455567294) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.001062913437441598;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                      result[0] += -0.0006728613236172332;
                    } else {
                      result[0] += -0.12130218160691353;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.034718038177541655;
                    } else {
                      result[0] += 0.006336868639411643;
                    }
                  }
                } else {
                  result[0] += -0.02710474608462169;
                }
              } else {
                result[0] += 0.0072592392250122725;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.0636020548757444;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0016207148642124886;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                    result[0] += 0.02340606358600778;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.006217152254990101;
                    } else {
                      result[0] += -0.023515989102666418;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
              result[0] += -0.035036635372588804;
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.031615248992806384;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += -0.018467635229538572;
                } else {
                  result[0] += 0.05315485150256527;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.0032696516051103075;
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += 0.033184633593536;
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.016957594931193427;
            } else {
              result[0] += -0.030101054280500444;
            }
          }
        } else {
          result[0] += -0.004734616241010549;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.068990230560303623) ) ) {
                      result[0] += -0.0056167379715112336;
                    } else {
                      result[0] += -0.02965295231559617;
                    }
                  } else {
                    result[0] += 8.426202869887906e-05;
                  }
                } else {
                  result[0] += 0.008430684296936749;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.07858228683471857) ) ) {
                    result[0] += 0.006619700257274193;
                  } else {
                    result[0] += -0.07450971065489354;
                  }
                } else {
                  result[0] += 0.041354537426517124;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += -0.08826111889310523;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81842756271362482) ) ) {
                  result[0] += -0.0038921722704066022;
                } else {
                  result[0] += -0.02600090169214805;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.01687710383721788;
            } else {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.03683163992072338;
              } else {
                result[0] += 0.10905628129396433;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.013688783674439639;
              } else {
                result[0] += -0.06361257202922072;
              }
            } else {
              result[0] += 0.008348280916584287;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.02446651458740412) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08102496430829063;
                  } else {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.021931778343447745;
                    } else {
                      result[0] += 0.07794722997030462;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.614335536956787998) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.45749855041504084) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.494223117828370029) ) ) {
                          result[0] += -0.03301718810404244;
                        } else {
                          result[0] += 0.12909281328087785;
                        }
                      } else {
                        result[0] += -0.13905971573662287;
                      }
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += 1.582037306273967e-05;
                        } else {
                          result[0] += -0.025423066675633002;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.45749855041504084) ) ) {
                          result[0] += 0.022170008989641648;
                        } else {
                          result[0] += 0.10225549638957086;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
                      result[0] += 0.006945141384921702;
                    } else {
                      result[0] += 0.02210298111195663;
                    }
                  }
                }
              } else {
                result[0] += 0.03782817237188811;
              }
            } else {
              result[0] += 0.030579770204988484;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.0013887326956462081;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.08435537568488793;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.053380152688968674;
            } else {
              result[0] += 0.07497534865711845;
            }
          }
        } else {
          result[0] += 0.0618998025373903;
        }
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
            result[0] += -0.002135906982824181;
          } else {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.030792676478954634;
            } else {
              result[0] += 0.00138105287962699;
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.015333238553656682;
            } else {
              result[0] += -0.01274661009180294;
            }
          } else {
            result[0] += -0.035611855040394394;
          }
        }
      } else {
        result[0] += -0.008842908925679014;
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.140846252441408026) ) ) {
      result[0] += -0.0004934759783919726;
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.004032283187079041;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.665476083755494052) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += 0.0075721105423998784;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                  result[0] += -0.010615132287070875;
                } else {
                  result[0] += -0.05979619568333005;
                }
              } else {
                if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.00048146136924343745;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.004400834280185928;
                  } else {
                    result[0] += 0.0979677655784237;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.009043596614569221;
            } else {
              result[0] += -0.054468007766320714;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.02262596250077543;
              } else {
                result[0] += -0.0304571250687063;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.68866753578186124) ) ) {
                      result[0] += 0.0023143372619263444;
                    } else {
                      result[0] += -0.10661428679239587;
                    }
                  } else {
                    result[0] += 0.05678077519835304;
                  }
                } else {
                  result[0] += 0.053217697739516207;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                    result[0] += -0.033606276305088716;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.01365498741376493;
                      } else {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.10912133423572186;
                        } else {
                          result[0] += 0.0011732261993227882;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.78399753570556818) ) ) {
                        result[0] += -0.02826617232718914;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += -0.040718588260886396;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.043741123787916794;
                          } else {
                            result[0] += 0.07993267836277684;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += 0.0032158266624240416;
                      } else {
                        result[0] += -0.044412378535030966;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.08580199149999218;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += -0.04653035776641232;
                        } else {
                          result[0] += 0.08303509049919239;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0766334533691424) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += -0.03564680576107894;
                      } else {
                        result[0] += 0.00708841289358292;
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.025992019629536695;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                                result[0] += -0.019296792732096613;
                              } else {
                                result[0] += -0.18812366008451234;
                              }
                            } else {
                              result[0] += 0.09611130828802099;
                            }
                          } else {
                            result[0] += 0.04781550081948647;
                          }
                        } else {
                          result[0] += 0.08395049855345216;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.019634532457520143;
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.014220966310064784;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                  result[0] += -0.1179489844689152;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.90263271331787287) ) ) {
                    result[0] += 0.018308828940955126;
                  } else {
                    result[0] += -0.13293514742576804;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.00198766307262127;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += -0.0011928911434214782;
          } else {
            result[0] += -0.017867294423177384;
          }
        }
      } else {
        if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.012605037751605068;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0007471376456300236;
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.010204768376403262;
            } else {
              result[0] += -0.024830630325266927;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.04900907547043933;
          } else {
            result[0] += -0.0033042591187217394;
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.736135363578796831) ) ) {
                result[0] += -0.014359717669799155;
              } else {
                result[0] += 0.022132914801964106;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.036739078035600066;
                  } else {
                    result[0] += 0.006304166826761973;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    result[0] += -0.01821166373223468;
                  } else {
                    result[0] += 0.02868237832568281;
                  }
                }
              } else {
                result[0] += 0.010258768968444543;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.04657301695191543;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += 0.010558889915454554;
              } else {
                result[0] += -0.04016597496142682;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.005050147030766744;
        } else {
          result[0] += -0.026225957269147244;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.29409265518188654) ) ) {
            result[0] += -0.0031742582586772312;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.052959703957719644;
              } else {
                result[0] += -0.03531484048525827;
              }
            } else {
              result[0] += 0.08807766650256556;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.680161952972413886) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              result[0] += 0.13352950329317323;
            } else {
              result[0] += -0.0063759556049588694;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.013490944590727583;
            } else {
              result[0] += -0.06810820139825523;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.83122920989990412) ) ) {
          result[0] += 0.01391937507420086;
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.39047479629516779) ) ) {
              result[0] += 0.031082354011025194;
            } else {
              result[0] += -0.1586515505547495;
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.018291151879748016;
            } else {
              result[0] += 0.12548112792027724;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.06573787546199507;
                  } else {
                    result[0] += -0.019724556516865176;
                  }
                } else {
                  result[0] += 0.019731322960902947;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += 0.04397721072999909;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.0854604627771714;
                  } else {
                    result[0] += -0.24064690546078515;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.009878507263372478;
              } else {
                result[0] += -0.11230263168755322;
              }
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += -0.014967829018278917;
            } else {
              result[0] += 0.0895332264123469;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
              result[0] += -0.07040851107639265;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += 0.05603307139717722;
              } else {
                result[0] += -0.03798154834644296;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.03859648153509697;
                } else {
                  result[0] += -0.06845859082215852;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                      result[0] += 0.14492080994214826;
                    } else {
                      result[0] += 0.011400097540296389;
                    }
                  } else {
                    result[0] += -0.07199276791948803;
                  }
                } else {
                  result[0] += 0.1078681674569983;
                }
              }
            } else {
              result[0] += -0.024983466275699195;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.006322684875323014;
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += 0.0059343694497990054;
                    } else {
                      result[0] += 0.10830430140356176;
                    }
                  }
                } else {
                  result[0] += -0.04238861068393601;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.03905095804603888;
                  } else {
                    result[0] += 0.038289115957877255;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.006729442758896448;
                  } else {
                    result[0] += 0.06143406487808395;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.08112854370264053;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.055496215820313388) ) ) {
                  result[0] += -0.04440485598796739;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.03219192790561286;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.002576828002931464) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                        result[0] += -0.09999176577016089;
                      } else {
                        result[0] += 0.058755153062088984;
                      }
                    } else {
                      result[0] += 0.12335665150572897;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.040752739887682486;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += 0.009301533671531212;
                } else {
                  result[0] += -0.018399823905408598;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.500490188598633701) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                      result[0] += -0.0029460933306171833;
                    } else {
                      result[0] += 0.0387069380234648;
                    }
                  } else {
                    result[0] += -0.05490559599316339;
                  }
                } else {
                  result[0] += 0.08057754172518773;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.725216388702393466) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                result[0] += 0.12249837985682606;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.005070827220837529;
                  } else {
                    result[0] += -0.1232736069662433;
                  }
                } else {
                  result[0] += 0.06605341834806444;
                }
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.636572122573853427) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.584475994110109198) ) ) {
                  result[0] += -0.08285146528772842;
                } else {
                  result[0] += 0.02343729026993014;
                }
              } else {
                result[0] += -0.2109503209548266;
              }
            }
          } else {
            result[0] += 0.02510253496319433;
          }
        }
      }
    }
  } else {
    result[0] += -0.00017311344704730387;
  }
  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
        result[0] += -0.003998521300218277;
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
              result[0] += -0.01892566433705251;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.10180357503391071;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.02996012626577766;
                  } else {
                    result[0] += 0.043401849416336964;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.445957899093628818) ) ) {
                    result[0] += 0.045571492807038955;
                  } else {
                    result[0] += -0.03648079686119833;
                  }
                } else {
                  result[0] += 0.00916678654112922;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18980646133423029) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                  result[0] += -0.09391916022012484;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.450390577316285068) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.08641886794342196;
                          } else {
                            result[0] += -0.0724539632140802;
                          }
                        } else {
                          result[0] += -0.1103733362801904;
                        }
                      } else {
                        result[0] += 0.04280848937300391;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.14646087838579466;
                        } else {
                          result[0] += 0.030527469954646787;
                        }
                      } else {
                        result[0] += 0.10550472376375951;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.070700883865357333) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                            result[0] += -0.10485674106923493;
                          } else {
                            result[0] += -0.003461992577840511;
                          }
                        } else {
                          result[0] += -0.1546949926730389;
                        }
                      } else {
                        result[0] += 0.05107462070086676;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                        result[0] += 0.10557461658878556;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                          result[0] += 0.0018119968844133906;
                        } else {
                          result[0] += 0.10287891624096664;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                  result[0] += 0.037544693912725184;
                } else {
                  result[0] += 0.1793274010343623;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  result[0] += -0.023549456913603817;
                } else {
                  result[0] += 0.11412313764558264;
                }
              } else {
                result[0] += -0.03348058290404083;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.016978855405614387;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.0701017815884723;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.10058788824565475;
                } else {
                  result[0] += 0.05826905475256747;
                }
              }
            }
          } else {
            result[0] += 0.081267268859791;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.04329792845752185;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12191963195800959) ) ) {
              result[0] += 0.002216821357719779;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12629127502441584) ) ) {
                result[0] += 0.12541840481430508;
              } else {
                result[0] += 0.026411225718996536;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.184114694595337802) ) ) {
              result[0] += 0.002298099043765074;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.16382805979393178;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.808102607727051669) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                            result[0] += 0.12553885265142914;
                          } else {
                            result[0] += -0.015711463802397806;
                          }
                        } else {
                          result[0] += -0.050470270513065865;
                        }
                      } else {
                        result[0] += 0.04112038245208881;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                              result[0] += -0.10052785998901691;
                            } else {
                              result[0] += 0.020897738112112235;
                            }
                          } else {
                            result[0] += -0.15591208227134518;
                          }
                        } else {
                          result[0] += -0.022736805003793208;
                        }
                      } else {
                        result[0] += -0.017613874729795167;
                      }
                    }
                  } else {
                    result[0] += -0.0016722941698284295;
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.85118436813354581) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.595096826553345615) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.009239995602780413;
                      } else {
                        result[0] += 0.007779328644121727;
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12348651885986506) ) ) {
                          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.13603348409431834;
                          } else {
                            result[0] += 0.15005609596157093;
                          }
                        } else {
                          result[0] += -0.12150657557033628;
                        }
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.018866985406970307;
                        } else {
                          result[0] += 0.08440950519764016;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.026179831966868256;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
            result[0] += -0.004113885923976969;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
              result[0] += 0.021485179964369338;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.537947177886963779) ) ) {
                result[0] += -0.26916796893406447;
              } else {
                result[0] += -0.07628979087015858;
              }
            }
          }
        } else {
          result[0] += -0.18755413727660072;
        }
      }
    }
  } else {
    result[0] += -0.00014426398612967444;
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.140846252441408026) ) ) {
      result[0] += -0.0005053420228454274;
    } else {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0029390322771279983;
            } else {
              result[0] += 0.017434649779726242;
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.76642942428588956) ) ) {
              result[0] += -0.02500606643854743;
            } else {
              result[0] += -0.13943099397953082;
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
              result[0] += 0.014718891746887503;
            } else {
              result[0] += -0.005237237394806478;
            }
          } else {
            result[0] += -0.006040723481534469;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.78704452514648615) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.07305450764781807;
              } else {
                result[0] += -0.012013544580316551;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
                  result[0] += -0.01161364432326397;
                } else {
                  result[0] += 0.010260627125237246;
                }
              } else {
                result[0] += -0.006587521412004885;
              }
            }
          } else {
            result[0] += 0.010553100331279144;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.011798029697186772;
          } else {
            result[0] += 0.0035539841594051744;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.0009481616690062636;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += -0.13055327763661045;
                } else {
                  result[0] += 0.06379720942019979;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                  result[0] += 0.012212164442510599;
                } else {
                  result[0] += -0.022874965997568553;
                }
              }
            }
          } else {
            result[0] += 0.021919446157288684;
          }
        } else {
          result[0] += -0.030459007140248268;
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
              result[0] += 0.019019559370880592;
            } else {
              result[0] += -0.01593404697730563;
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0013837939436176698;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.011045854656916773;
              } else {
                result[0] += -0.015518063003324853;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
            result[0] += -0.002986810503045497;
          } else {
            result[0] += -0.02755826118858648;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
        result[0] += 0.002840140921161135;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += 0.0024139755664495492;
                      } else {
                        result[0] += -0.02693709876852936;
                      }
                    } else {
                      result[0] += 0.05012386111977324;
                    }
                  } else {
                    result[0] += -0.03848491453382001;
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += 0.024712686774020783;
                    } else {
                      result[0] += 0.002803246313358222;
                    }
                  } else {
                    result[0] += -0.014138974203885905;
                  }
                }
              } else {
                result[0] += 0.07364962695098419;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.005072760867014915;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.02917056238991684;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.03760759370122155;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.184114694595337802) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.007298471455006486;
                        } else {
                          result[0] += -0.04245014927125026;
                        }
                      } else {
                        result[0] += 0.013858250788613486;
                      }
                    }
                  }
                } else {
                  result[0] += 0.021895298856327794;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                result[0] += -0.005806931676389477;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.010405664989949569;
                } else {
                  result[0] += 0.016567390880276332;
                }
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.00013161616649100025;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                          result[0] += -0.04730704576249498;
                        } else {
                          result[0] += 0.06217745501450808;
                        }
                      } else {
                        result[0] += 0.013712256396855853;
                      }
                    }
                  } else {
                    result[0] += -0.01169046518619021;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += 0.07035109385577483;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
                      result[0] += 0.011406599534093896;
                    } else {
                      result[0] += 0.1476694057205947;
                    }
                  }
                }
              } else {
                result[0] += -0.0033047718415508045;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
            result[0] += -0.02339339334487581;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += 0.020991626135854775;
              } else {
                result[0] += -0.020214355983135125;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.044356851080524624;
              } else {
                result[0] += -0.024774070068043247;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.020600522095433033;
        } else {
          result[0] += -0.004110861711888677;
        }
      } else {
        result[0] += -0.00048052428211913455;
      }
    } else {
      result[0] += 0.0018464450893190501;
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
              result[0] += -0.0782843980641432;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += 0.07950331492996576;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                      result[0] += 0.0721205012543282;
                    } else {
                      result[0] += -0.0033319844960938735;
                    }
                  } else {
                    result[0] += -0.011110589478648524;
                  }
                } else {
                  result[0] += -0.03619729863469579;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.963667154312134677) ) ) {
              result[0] += 0.014352202754794402;
            } else {
              result[0] += -0.10113248854895776;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += 0.017292389582310035;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.020526716953222175;
                } else {
                  result[0] += -0.06313693581816497;
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += -0.0005519795136552985;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.017203195114385186;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
                      result[0] += -0.08065183134330045;
                    } else {
                      result[0] += 0.05454633237631318;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += -0.04165343868534011;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                      result[0] += 0.005884190571403762;
                    } else {
                      result[0] += -0.10051061679370772;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                    result[0] += 0.01850040821741948;
                  } else {
                    result[0] += 0.07790463552405742;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += 0.008406160016062786;
                } else {
                  result[0] += -0.014502615572204053;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += 0.041273503666319794;
                  } else {
                    result[0] += 0.006731376990102087;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.027152359776592666;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.04703678998025739;
                    } else {
                      result[0] += 0.10058420586134292;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += 0.07258248232869828;
              } else {
                result[0] += -0.04848939501174123;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.01903647334906911;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.11829767865679307;
          } else {
            result[0] += 0.03334671936683006;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
        result[0] += -0.0004248542189034371;
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.795426130294800249) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.957021474838257724) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10223627090454279) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += -0.040143036596087596;
                    } else {
                      result[0] += 0.07379043825388719;
                    }
                  } else {
                    result[0] += -0.09178855265627985;
                  }
                } else {
                  result[0] += 0.13905589020097792;
                }
              } else {
                result[0] += 0.1380950538273232;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.017375230789185458) ) ) {
                result[0] += -0.16426797598964418;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11247110366821467) ) ) {
                  result[0] += -0.1114815417426992;
                } else {
                  result[0] += 0.022701120390796903;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12222146987915217) ) ) {
                result[0] += -0.10500398221495388;
              } else {
                result[0] += 0.08905217167768424;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.12963804581037758;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12553024291992365) ) ) {
                  result[0] += 0.11504263694257676;
                } else {
                  result[0] += -0.09213502152000018;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.005734163797126729;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += -0.006380887803915023;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.970467567443849433) ) ) {
                  result[0] += -0.030657204562270314;
                } else {
                  result[0] += -0.22512375605700155;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                result[0] += -0.024751539546622466;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.35697507858276456) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.448499202728272373) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.05097036014625494;
                      } else {
                        result[0] += 0.1372389845395006;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.658699750900269443) ) ) {
                        result[0] += -0.15931099027169648;
                      } else {
                        result[0] += 0.03138959944654229;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.370756626129151279) ) ) {
                      result[0] += -0.21493673925842663;
                    } else {
                      result[0] += -0.048671827479178836;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.06107273925287651;
                  } else {
                    result[0] += -0.005928700840290623;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
      result[0] += 0.0007148377104655346;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.30350399017334162) ) ) {
        result[0] += -0.06597213550252551;
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.04733747904072731;
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.005975448024551671;
          } else {
            result[0] += 0.13497024750301112;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += 0.000491806856190666;
        } else {
          result[0] += -0.024051771954494497;
        }
      } else {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.08144326231787571;
                  } else {
                    result[0] += -0.01233783286996121;
                  }
                } else {
                  result[0] += -0.013157041568496692;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.011169020786061357;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                    result[0] += -0.004213106640791006;
                  } else {
                    result[0] += 0.012393649944406341;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0009483687238704303;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
                      result[0] += -0.015687207197044892;
                    } else {
                      result[0] += 0.06550514868640318;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.024311247439544333;
                      } else {
                        result[0] += -0.13217008097396615;
                      }
                    } else {
                      result[0] += 0.03549704685723794;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                      result[0] += 0.07392644747309364;
                    } else {
                      result[0] += 0.025310483684325105;
                    }
                  } else {
                    result[0] += 0.015866243295950128;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    result[0] += 0.10997622273063751;
                  } else {
                    result[0] += -0.06026147940343868;
                  }
                } else {
                  result[0] += 0.01663838133491343;
                }
              } else {
                result[0] += -0.025112727310499506;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.08597840336155937;
                  } else {
                    result[0] += 0.008572941601048068;
                  }
                } else {
                  result[0] += -0.013826436468977122;
                }
              } else {
                result[0] += 0.05090463057319412;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += -0.022067281929641677;
              } else {
                result[0] += 0.02574015939420555;
              }
            } else {
              result[0] += -0.0028317985759675488;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
              result[0] += 0.022999331309455282;
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.11546548179177568;
                } else {
                  result[0] += -0.02572886491520811;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.03647567436261933;
                  } else {
                    result[0] += -0.0022849299554517275;
                  }
                } else {
                  result[0] += 0.1078110840665936;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.01634240150451749) ) ) {
              result[0] += 0.0632702309749716;
            } else {
              result[0] += 0.011468267416721151;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.166635274887085849) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.012716068394764114;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                          result[0] += 0.013121620210908975;
                        } else {
                          result[0] += 0.15800366835312857;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.11650121678464703;
                      } else {
                        result[0] += 0.7532776479517658;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.009941972894809643;
                    } else {
                      result[0] += -0.02658340912079627;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0033224985185215843;
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.008139437993111684;
                    } else {
                      result[0] += -0.04502277004585859;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                  result[0] += -0.021382592142687302;
                } else {
                  result[0] += 0.006767293813424326;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.01743401929623903;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                  result[0] += -0.02368193205291522;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.07479016715270981;
                  } else {
                    result[0] += 0.004906774110404349;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.024178901534962285;
              } else {
                result[0] += -0.0015748532363727232;
              }
            } else {
              result[0] += -0.03256499886590557;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -9.104815827992826e-06;
            } else {
              result[0] += -0.060176620850616064;
            }
          }
        }
      } else {
        result[0] += -0.032984713540752546;
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
      result[0] += 0.0007397016455633231;
    } else {
      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.14520875942928868;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.01867473065159102;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.78711032867431818) ) ) {
                      result[0] += -0.0038987464782335255;
                    } else {
                      result[0] += -0.10983061254866876;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                    result[0] += -0.14076160856364328;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.028772890349337606;
                    } else {
                      result[0] += 0.0064548792822012695;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.05091281467646829;
                } else {
                  result[0] += -0.06033265044853368;
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                    result[0] += 0.002239134522789648;
                  } else {
                    result[0] += -0.03059726132864554;
                  }
                } else {
                  result[0] += 0.012351984995950626;
                }
              } else {
                result[0] += 0.014866870160279058;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.799612998962403232) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0006227914964531988;
                } else {
                  result[0] += -0.07972444300577564;
                }
              } else {
                result[0] += 0.02008308235049031;
              }
            } else {
              result[0] += 0.1334989791455866;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += 0.07403153414543219;
            } else {
              result[0] += -0.04256343542607577;
            }
          } else {
            result[0] += -0.12085005864213372;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
              result[0] += 0.03199928488465167;
            } else {
              result[0] += -0.04602156237091554;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.05729676613955298;
            } else {
              result[0] += -0.012896339434768254;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.004383908230118237;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += -0.004643614854122067;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.021433956212659953;
                    } else {
                      result[0] += 0.0773356183663395;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.04020631780068834;
                    } else {
                      result[0] += 0.019329567172000375;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                  result[0] += 0.011095828340896426;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += 0.05815111711049184;
                  } else {
                    result[0] += 0.012868601526011929;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
              result[0] += 0.028798989735767135;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.01442433939225887;
              } else {
                result[0] += -0.05525877432677188;
              }
            }
          }
        } else {
          result[0] += 0.01522379708050076;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += 0.03024137514590397;
            } else {
              result[0] += -0.00340677404543379;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += 0.021985591605696894;
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.025079651910687402;
                } else {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.1027813831859888;
                  } else {
                    result[0] += -0.006633267179998275;
                  }
                }
              }
            } else {
              result[0] += 0.03128870001450483;
            }
          }
        } else {
          result[0] += -0.005429080780914181;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
            result[0] += 0.025900185848476993;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
              result[0] += -0.008227520598010659;
            } else {
              result[0] += 0.0368313406771549;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.004983582511140369;
              } else {
                result[0] += -0.06350558441120595;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                    result[0] += 0.06660278414472585;
                  } else {
                    result[0] += -0.05849504549436412;
                  }
                } else {
                  result[0] += -0.015979672067043312;
                }
              } else {
                result[0] += -0.03206727774545313;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
                result[0] += -0.0019796988801979807;
              } else {
                result[0] += -0.012328080527519075;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.021419351115509907;
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.026156572844797985;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                      result[0] += 0.02705941726985967;
                    } else {
                      result[0] += 0.09475653099655298;
                    }
                  }
                } else {
                  result[0] += -0.005677441483042156;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.03502716946172675;
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.00046202779075233536;
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.003144139312178842;
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
            result[0] += 0.004621798721738545;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.024984851813644657;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.07833683159053663;
                  } else {
                    result[0] += 0.0923102651875046;
                  }
                }
              } else {
                result[0] += 0.016692836114143166;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.017450309125360313;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                      result[0] += 0.09046922766656462;
                    } else {
                      result[0] += -0.070110787082043;
                    }
                  }
                } else {
                  result[0] += 0.04250384583600005;
                }
              } else {
                result[0] += -0.024529641026511437;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.017965171426608376;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
            result[0] += 0.0005030454038854928;
          } else {
            result[0] += 0.06262076981885235;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.002869745405299184;
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.049912098063834814;
              } else {
                result[0] += 0.0012626050110218449;
              }
            } else {
              result[0] += 0.002751948891287305;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.03272492453102726;
            } else {
              result[0] += -0.015220425654026065;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += -0.02889112644970468;
              } else {
                result[0] += 0.0906807883728118;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.497398376464845526) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                        result[0] += 0.09039355437484725;
                      } else {
                        result[0] += -0.09097520540999615;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                        result[0] += -0.17283989439308703;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                          result[0] += -0.08337249633585626;
                        } else {
                          result[0] += 0.061298225756568986;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.03845697161202892;
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                    result[0] += -0.00794320471373099;
                  } else {
                    result[0] += -0.17591128418606505;
                  }
                }
              } else {
                result[0] += -0.002577949887507913;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
          result[0] += 0.0013384951135374072;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.30013217947994586;
                  } else {
                    result[0] += -0.016523958796068154;
                  }
                } else {
                  result[0] += 0.007356486279640816;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.73821687698364435) ) ) {
                  result[0] += 0.02395963727056907;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.11543857968113967;
                  } else {
                    result[0] += -0.0014807653347194165;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.486867427825928623) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.11367801356082992;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.014821585082571729;
                      } else {
                        result[0] += 0.10091482401718417;
                      }
                    } else {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                        result[0] += 0.004243482693065802;
                      } else {
                        result[0] += 0.1426538620908397;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12428855895996271) ) ) {
                      result[0] += 0.022190541422433385;
                    } else {
                      result[0] += -0.02134881589772404;
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.953768730163575995) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.692703723907471591) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.648983716964722568) ) ) {
                          result[0] += -0.03218479246230922;
                        } else {
                          result[0] += -0.1859011627324787;
                        }
                      } else {
                        result[0] += 0.02856898842730035;
                      }
                    } else {
                      result[0] += -0.13353412039099913;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                  result[0] += 0.03880747483578617;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.2736524669742734;
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.09606038540761867;
                    } else {
                      result[0] += 0.0009741810560387391;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801954269409180576) ) ) {
                    result[0] += -0.012239375986523202;
                  } else {
                    result[0] += 0.06332991325867042;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03866420460198561;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.04436939987382013;
                    } else {
                      result[0] += 0.034907818144738846;
                    }
                  }
                }
              } else {
                result[0] += -0.02624563917649655;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += 0.06948162473458669;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.005778754746829363;
                } else {
                  result[0] += -0.054670133755734596;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.00037723140409374166;
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.005964550584557397;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            result[0] += 0.006582268171536895;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.11496930545374998;
            } else {
              result[0] += -0.022785817705384664;
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.052787423456691296;
          } else {
            result[0] += 0.017561625252901603;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.777674019336700661) ) ) {
        result[0] += 0.1919855374004692;
      } else {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.591613531112671787) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.03190704700025098;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                      result[0] += -0.013232590276215728;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                        result[0] += 0.016814415519386526;
                      } else {
                        result[0] += 0.17473137795790294;
                      }
                    }
                  } else {
                    result[0] += 0.01324353386740049;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                        result[0] += -0.005263659934132749;
                      } else {
                        result[0] += 0.002754426167370823;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += 0.05175978101430548;
                      } else {
                        result[0] += 0.002756238498064667;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.023953641954863044;
                      } else {
                        result[0] += 0.053731505968848584;
                      }
                    } else {
                      result[0] += 0.0059392253429707415;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += -0.018690448975034298;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.33055067062378107) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                              result[0] += -0.004880836469350083;
                            } else {
                              result[0] += -0.09192401882360099;
                            }
                          } else {
                            result[0] += 0.076065532651117;
                          }
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.025902450355085444;
                          } else {
                            result[0] += -0.1375025257138903;
                          }
                        }
                      } else {
                        result[0] += -0.13298703346613275;
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.10402076394615943;
                      } else {
                        result[0] += -3.786479937384533e-05;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                        result[0] += 0.08383124874147;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                          result[0] += 0.04850301309745414;
                        } else {
                          result[0] += -0.07254688654139534;
                        }
                      }
                    } else {
                      result[0] += 0.03140600265312198;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                result[0] += -0.09280088699540677;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.676220536231995073) ) ) {
                  result[0] += 0.12404708146806623;
                } else {
                  result[0] += 0.0191117967571236;
                }
              }
            } else {
              result[0] += -0.003283758575806709;
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.015324725093850395;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.08243318811825688;
              } else {
                result[0] += 0.02385667089956371;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += 0.12158433656513706;
                      } else {
                        result[0] += 0.060700793178361405;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                        result[0] += 0.19130760677283842;
                      } else {
                        result[0] += -0.01735574226835161;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                            result[0] += -0.12990978666179315;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10889482498169123) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.04641723632812678) ) ) {
                                result[0] += 0.04629814833449123;
                              } else {
                                result[0] += 0.18326352852691719;
                              }
                            } else {
                              result[0] += -0.026537660798207765;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.000000000000000888) ) ) {
                            result[0] += -0.038476385606067864;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += -0.03716240485680393;
                            } else {
                              result[0] += 0.02569965161480472;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                          result[0] += 0.03220067475617795;
                        } else {
                          result[0] += -0.04476098091620295;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.361115694046021396) ) ) {
                        result[0] += 0.005088375990359236;
                      } else {
                        result[0] += 0.019405658603543534;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.05241199738937016;
                  } else {
                    result[0] += -0.002221135946845855;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.14509192171801322;
                      } else {
                        result[0] += 0.005402950849950488;
                      }
                    } else {
                      result[0] += -1.198398678415224e-05;
                    }
                  } else {
                    result[0] += -0.03064704881736198;
                  }
                } else {
                  result[0] += 0.08825027162585494;
                }
              }
            } else {
              result[0] += -0.018625185631267912;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.00022145484949202688;
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.61744737625122248) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
          result[0] += -0.0019688036298740673;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.992907285690308505) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.043341273844284364;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.158509254455567294) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                          result[0] += -0.02590539432936398;
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += -0.11490273532260563;
                          } else {
                            result[0] += -0.02574273286511715;
                          }
                        }
                      } else {
                        result[0] += 0.006954813273096688;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += 0.02018367259053342;
                      } else {
                        result[0] += -0.004062584826311939;
                      }
                    }
                  } else {
                    result[0] += 0.01830691495047406;
                  }
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                      result[0] += 0.053487754984843885;
                    } else {
                      result[0] += 0.013354804848070527;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                        result[0] += -0.025033104957480096;
                      } else {
                        result[0] += 0.024492060495990012;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.017430535959050298;
                      } else {
                        result[0] += -0.03052928159304613;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.016903589858631003;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.552201986312867099) ) ) {
                result[0] += 0.020119247537309416;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  result[0] += 0.013168680470378774;
                } else {
                  result[0] += -0.03732532720254942;
                }
              }
            } else {
              result[0] += -0.03585944479511903;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.15307623056820496;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.791641235351563388) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.016853765659994507;
                    } else {
                      result[0] += 0.021892252609706148;
                    }
                  } else {
                    result[0] += 0.048229993641862534;
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
                    result[0] += 0.00804123328872213;
                  } else {
                    result[0] += 0.041602748467697466;
                  }
                } else {
                  result[0] += -0.08294189406951596;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.0034825149207756098;
                  } else {
                    result[0] += 0.04513590182717838;
                  }
                } else {
                  result[0] += -0.01505431638282568;
                }
              } else {
                result[0] += 0.00791755874623262;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              result[0] += -0.036703996946877114;
            } else {
              result[0] += -0.0050154309100343165;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.572941064834595615) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.5431861877441424) ) ) {
                  result[0] += -0.009202431425184517;
                } else {
                  result[0] += 0.13917429856865474;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.09662018725690746;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.658699750900269443) ) ) {
                        result[0] += 0.01731961913733446;
                      } else {
                        result[0] += -0.01745975541570227;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                        result[0] += -0.025879880455349905;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                              result[0] += 0.08497235677364903;
                            } else {
                              result[0] += -0.00041545523504277595;
                            }
                          } else {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                              result[0] += 0.08931025371932153;
                            } else {
                              result[0] += -0.03923480277071094;
                            }
                          }
                        } else {
                          result[0] += 0.0373457105097265;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.992907285690308505) ) ) {
                          result[0] += 0.011602600814057683;
                        } else {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.017695051843919025;
                          } else {
                            result[0] += 0.18196508249335375;
                          }
                        }
                      } else {
                        result[0] += -0.009906028756398204;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.00011807289742026676;
                  } else {
                    result[0] += 0.09553499628817222;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.014075587510821426;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.006070020998322698;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                          result[0] += -0.022873693912810876;
                        } else {
                          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.06866525119098271;
                          } else {
                            result[0] += 0.005275079108020234;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0470556372235132;
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.03161622824325311;
                      } else {
                        result[0] += 0.016523960285469996;
                      }
                    }
                  }
                } else {
                  result[0] += -0.029547079182380066;
                }
              } else {
                result[0] += 0.07387878540778878;
              }
            }
          } else {
            result[0] += -0.02621654519753029;
          }
        }
      }
    } else {
      result[0] += -0.018313109805866676;
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.00047183274127504736;
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.022655167499138006;
        } else {
          result[0] += 0.004479809123080744;
        }
      } else {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0010918152799702693;
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.012479866266711062;
            } else {
              result[0] += 0.024091229359010514;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.00667320033739522;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.0014754082755846997;
                    } else {
                      result[0] += 0.03141699722996553;
                    }
                  } else {
                    result[0] += -0.006564946658232863;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                  result[0] += -0.004719561046773679;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0010306430520512226;
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.09431384459623758;
                    } else {
                      result[0] += 0.05377426997033126;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.01493010784600759;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.04004337748201431;
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += 0.04135925556787221;
                  } else {
                    result[0] += -0.021593949466935013;
                  }
                } else {
                  result[0] += 0.08343402783105779;
                }
              }
            } else {
              result[0] += -0.03144534100276608;
            }
          } else {
            result[0] += -0.041480874621138794;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
              result[0] += -0.020237519615277942;
            } else {
              result[0] += 0.07410778958512658;
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.166635274887085849) ) ) {
              result[0] += 0.06592146182364376;
            } else {
              result[0] += 0.016767664265737208;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
          result[0] += -0.0454902855614468;
        } else {
          result[0] += 0.10805814152765925;
        }
      }
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.06431262889129329;
            } else {
              result[0] += 0.00955994236937778;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.0036956006596075342;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.1567498622753135;
              } else {
                result[0] += 0.0747273059975475;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.774546623229981357) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.363266706466675693) ) ) {
                  result[0] += 0.018235185301930718;
                } else {
                  result[0] += -0.026226729749691938;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += 0.030211171237544732;
                  } else {
                    result[0] += -0.007414308795755645;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.06646948298143758;
                  } else {
                    result[0] += 0.08102659083055613;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.0701037438615861;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += 0.08896328196208184;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += 0.029045131289798156;
                  } else {
                    result[0] += 0.2025830533034858;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0031026205674865074;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                        result[0] += -0.011941562932005552;
                      } else {
                        result[0] += -0.08877402603885348;
                      }
                    } else {
                      result[0] += -0.04822845356437435;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.026760001566382415;
                    } else {
                      result[0] += 0.19125080554387552;
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.024730086389555234;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.12276034807758487;
                        } else {
                          result[0] += 0.02093212543527724;
                        }
                      } else {
                        result[0] += -0.005441996521796947;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.05867651856633349;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += -0.02737109595301559;
              } else {
                result[0] += 0.07677212035955491;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.05598805837200036;
            } else {
              result[0] += -0.024206076151514308;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.228399038314820224) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.07594756755580295;
              } else {
                result[0] += -0.27829265905446093;
              }
            } else {
              result[0] += -0.04089824260757852;
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.01017723015211681;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.774546623229981357) ) ) {
              result[0] += 0.20229598985639288;
            } else {
              result[0] += 0.0701317385233347;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.1062313083109179;
          } else {
            result[0] += -0.004743256722233995;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.05546688815876402;
          } else {
            result[0] += -0.024984482018825888;
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += 0.0013164734586693949;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0005168637014043997;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                      result[0] += 0.0037853452285798923;
                    } else {
                      result[0] += -0.034925737308428496;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.023651430122605843;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                        result[0] += -0.03964734288637661;
                      } else {
                        result[0] += -0.09379588757261695;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0034935259053532966;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0732909365929631;
                  } else {
                    result[0] += -0.04929779981344001;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.01149515434196434;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += -0.08472827911034629;
                } else {
                  result[0] += 0.04211035393179409;
                }
              }
            }
          } else {
            result[0] += -0.037310643474146456;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.003232119566038994;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0369817478614333;
              } else {
                result[0] += -0.07370621497783027;
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920601367950440341) ) ) {
              result[0] += 0.09168816231576779;
            } else {
              result[0] += -0.043273554076984994;
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.0003815743126175396;
            } else {
              result[0] += 0.010843856815790798;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
              result[0] += -0.0045241650686575876;
            } else {
              result[0] += -0.03302743767931557;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
            result[0] += 0.045436831029331874;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.007847447503559472;
            } else {
              result[0] += -0.052015827605414615;
            }
          }
        } else {
          result[0] += 0.006150763703498197;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      result[0] += 0.0007212562791882693;
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += 0.025155495958955568;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.0017844309399572666;
              } else {
                result[0] += -0.022977239141711996;
              }
            }
          } else {
            result[0] += -0.04269144142651313;
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.012130171774061295;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
                  result[0] += -0.017461517889877994;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.024276324985038723;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.006383704751535789;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                          result[0] += -0.048351608277789344;
                        } else {
                          result[0] += 0.048055481854906726;
                        }
                      }
                    } else {
                      result[0] += 0.027233483820106097;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.944020271301270419) ) ) {
                    result[0] += -3.9737188720160866e-05;
                  } else {
                    result[0] += -0.06890335815630881;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.614335536956787998) ) ) {
                              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += -0.1585596974449313;
                              } else {
                                result[0] += 0.0031892025240650014;
                              }
                            } else {
                              result[0] += -0.06145188946966598;
                            }
                          } else {
                            result[0] += 0.049901777048146995;
                          }
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                              result[0] += -0.017369970819220783;
                            } else {
                              result[0] += -0.19112982533439202;
                            }
                          } else {
                            result[0] += 0.023039218998327254;
                          }
                        }
                      } else {
                        result[0] += 0.009056277564857301;
                      }
                    } else {
                      result[0] += 0.014135273917234176;
                    }
                  } else {
                    result[0] += 0.031107831163556;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.0020184223709351946;
            } else {
              result[0] += -0.030884215119175586;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.031532673423714574;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.0016740863816041892;
            } else {
              result[0] += 0.043628092162986604;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.94957673549652144) ) ) {
            result[0] += -0.01632679139453409;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.025192260742188388) ) ) {
              result[0] += 0.011358752686324323;
            } else {
              result[0] += -0.0061785279719869714;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.0005076609671072596;
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.011291937859238531;
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.58961367607116788) ) ) {
              result[0] += 0.020780169908295;
            } else {
              result[0] += -0.03660503381776994;
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.005090807324622058;
          } else {
            result[0] += -0.052918022649411416;
          }
        }
      } else {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.74006319046020685) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.036670446395874912) ) ) {
                result[0] += 0.018106846286090283;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
                    result[0] += 0.09288866872647492;
                  } else {
                    result[0] += -0.0513477980597121;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                    result[0] += 0.014285919364782221;
                  } else {
                    result[0] += -0.005715763516803222;
                  }
                }
              }
            } else {
              result[0] += -0.010824348235792612;
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.007542007647341875;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += 0.00724447597903427;
                        } else {
                          result[0] += 0.12533760483673764;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                          result[0] += -0.025997904160449833;
                        } else {
                          result[0] += -0.10210846955547499;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.105651378631592685) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.0025504265505497557;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.055496215820313388) ) ) {
                              result[0] += -0.03602988035400092;
                            } else {
                              result[0] += -0.005193616059870512;
                            }
                          }
                        } else {
                          result[0] += 0.005899643236297403;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.102609157562256748) ) ) {
                            result[0] += 0.01798035722959438;
                          } else {
                            result[0] += 0.055265073034482096;
                          }
                        } else {
                          result[0] += 0.001513505326470498;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.02502262441905806;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.593729019165039951) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += 0.03845665673940862;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.736968994140626776) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                              result[0] += -0.10302725742123525;
                            } else {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.500490188598633701) ) ) {
                                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.289595603942871982) ) ) {
                                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                                    result[0] += -0.03588771076222393;
                                  } else {
                                    result[0] += 0.016058524773495477;
                                  }
                                } else {
                                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                                    result[0] += -0.12024590638175475;
                                  } else {
                                    result[0] += -0.03419286488434862;
                                  }
                                }
                              } else {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                                  result[0] += 0.05773656183110332;
                                } else {
                                  result[0] += -0.05613465879352737;
                                }
                              }
                            }
                          } else {
                            result[0] += 0.008364275554094585;
                          }
                        } else {
                          result[0] += 0.07941814048218125;
                        }
                      }
                    } else {
                      result[0] += 0.01158062706600813;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.06580510308379366;
                    } else {
                      result[0] += 0.009915469963430928;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                    result[0] += 0.0057008986424808535;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                      result[0] += -0.015690725538760553;
                    } else {
                      result[0] += -0.0357341775671241;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.634540319442749912) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.00799570614437965;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                        result[0] += 0.03185867772902431;
                      } else {
                        result[0] += -0.014556293277105687;
                      }
                    } else {
                      result[0] += -0.0052024520761640705;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.0452756730331168;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.579839229583741123) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                          result[0] += -0.006701669967483701;
                        } else {
                          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.03873754880790872;
                          } else {
                            result[0] += -0.0130520991594071;
                          }
                        }
                      } else {
                        result[0] += 0.02181955039647154;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.00851866755961075;
                        } else {
                          result[0] += -0.03450321156310448;
                        }
                      } else {
                        result[0] += -0.00869942634628788;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.44140100479126021) ) ) {
                  result[0] += -0.015162871044794055;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0052445649147480795;
                  } else {
                    result[0] += 0.04028403594378051;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.01616966912506677;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.655405282974244052) ) ) {
                result[0] += 0.0026366912327743736;
              } else {
                result[0] += 0.06156993346981596;
              }
            } else {
              result[0] += -0.014503674387192368;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
        result[0] += -0.2371324343549881;
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.03626142758947628;
        } else {
          result[0] += -0.025732027888558308;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.005588297632476335;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.057648150795149095;
              } else {
                result[0] += 0.009414213742677188;
              }
            }
          } else {
            result[0] += -0.0052708999768264105;
          }
        } else {
          result[0] += -0.002791353524078817;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.09290879076053701;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0009644063107028261;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.25034575560787176;
                          } else {
                            result[0] += -0.030874135479519033;
                          }
                        } else {
                          result[0] += 0.001317454327025222;
                        }
                      } else {
                        result[0] += 0.008681905137958714;
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.81628036499023615) ) ) {
                        result[0] += -0.042738730786747156;
                      } else {
                        result[0] += 0.03711611794786643;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.659039497375490058) ) ) {
                        result[0] += -0.04440465303140678;
                      } else {
                        result[0] += 0.0005378978305163886;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                        result[0] += -0.033380069376520986;
                      } else {
                        result[0] += -0.09093108340466041;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.002924347337867406;
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.042766405172990796;
                  } else {
                    result[0] += -0.07875766092182895;
                  }
                } else {
                  result[0] += 0.09883851880189866;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.011354268091672602;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += 0.0717023589264371;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.14690954452484611;
                } else {
                  result[0] += -0.028552731170442953;
                }
              }
            }
          }
        } else {
          result[0] += -0.033851398415000594;
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8006772994995135) ) ) {
                result[0] += -0.023311450885769622;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.00764274597168146) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.10026099560327878;
                    } else {
                      result[0] += -0.04353720795327098;
                    }
                  } else {
                    result[0] += 0.08582218260507428;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.03192359816591748;
                  } else {
                    result[0] += -0.02210902492844534;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.028023371636861468;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.212100267410279208) ) ) {
                  result[0] += 0.003881831147093527;
                } else {
                  result[0] += -0.01814675162619489;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.007999110659289562;
            } else {
              result[0] += 0.02787719615618153;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.006318524380355878;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.55753517150879084) ) ) {
                result[0] += 0.0014716279429034052;
              } else {
                result[0] += 0.0332912542668709;
              }
            }
          } else {
            result[0] += 0.0014024522593743473;
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.02168593174224606;
          } else {
            result[0] += -0.002426831021580829;
          }
        } else {
          result[0] += 0.013588089086706818;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.01322700514274043;
              } else {
                result[0] += 0.03929778159656748;
              }
            } else {
              result[0] += 0.003106129224286584;
            }
          } else {
            result[0] += -0.025289315332146362;
          }
        } else {
          result[0] += -0.004496542931125193;
        }
      } else {
        result[0] += -0.00027850365949206506;
      }
    } else {
      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.0033312429486309922;
          } else {
            result[0] += 0.08954134497297378;
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.0004017627020152867;
            } else {
              result[0] += -0.030903967238067633;
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                  result[0] += 0.0009106960600318439;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.020723223487178366;
                  } else {
                    result[0] += -0.04903624451347061;
                  }
                }
              } else {
                result[0] += 0.012452355478999533;
              }
            } else {
              result[0] += 0.031591861364546316;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.00013782643582688065;
            } else {
              result[0] += -0.027668203629485674;
            }
          } else {
            result[0] += -0.00011660268719048973;
          }
        } else {
          result[0] += 0.013414223204107715;
        }
      }
    }
  }
  if ( UNLIKELY(  (data[32].missing != -1) && (data[32].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += -0.08300154818527827;
      } else {
        result[0] += 0.0023881554717018923;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
        result[0] += 0.18661491524063564;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.09524002044831446;
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02030224499636302;
                } else {
                  result[0] += -0.14819695998214236;
                }
              } else {
                result[0] += 0.0276161147525103;
              }
            } else {
              result[0] += 0.1325311327049927;
            }
          }
        } else {
          result[0] += 0.11990349803484623;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
      result[0] += 0.00019322295335112978;
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.09817006037179755;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.96495962142944514) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.72293281555175959) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.002731216506000984;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                      result[0] += 0.027151810853173836;
                    } else {
                      result[0] += 0.13985739412545048;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.07319864715830893;
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                        result[0] += 0.00134176865593965;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                          result[0] += -0.0656839916105405;
                        } else {
                          result[0] += 0.06844268310086908;
                        }
                      }
                    } else {
                      result[0] += -0.1402631746498782;
                    }
                  }
                }
              } else {
                result[0] += 0.09552106965196479;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.66117286682129084) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += -0.010803002078874885;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.055496215820313388) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                            result[0] += 0.055876136228809284;
                          } else {
                            result[0] += -0.06062935603470582;
                          }
                        } else {
                          result[0] += -0.12529579220561993;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                          result[0] += -0.029465766680103745;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.561121463775635654) ) ) {
                              result[0] += 0.07653055538496806;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                                result[0] += 0.08887124656085502;
                              } else {
                                result[0] += -0.02507125660784792;
                              }
                            }
                          } else {
                            result[0] += 0.11692437348447332;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                        result[0] += 0.08388061211424759;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42620229721069514) ) ) {
                              result[0] += -0.0660807959622066;
                            } else {
                              result[0] += 0.07038061316317318;
                            }
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                                result[0] += 0.09596058303301574;
                              } else {
                                result[0] += 0.026471044007027586;
                              }
                            } else {
                              result[0] += 0.1526378017680222;
                            }
                          }
                        } else {
                          result[0] += -0.03939243756792416;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.02853568537631092;
                }
              } else {
                result[0] += -0.09187798839541647;
              }
            }
          } else {
            if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.731793165206910068) ) ) {
                result[0] += -0.07978696230793515;
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += -0.05892116095153569;
                } else {
                  result[0] += 0.005041714133043913;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.23485788753164483;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.025780646463325743;
                  } else {
                    result[0] += 0.15216212980083593;
                  }
                }
              } else {
                result[0] += -0.015729139750488404;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += 0.006086083846534442;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                result[0] += -0.014407253804783538;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.215607166290284091) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += 0.04339117242972639;
                    } else {
                      result[0] += -0.0004524634807314341;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.289602279663086826) ) ) {
                      result[0] += 0.05398923777894992;
                    } else {
                      result[0] += 0.010978386074101098;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.26332664489746271) ) ) {
                      result[0] += -0.015435919027147374;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.002102981791472047;
                      } else {
                        result[0] += -0.17778358532447022;
                      }
                    }
                  } else {
                    result[0] += 0.15266745170240859;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.205894470214845526) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.640107631683351386) ) ) {
                        result[0] += 0.01213617919514897;
                      } else {
                        result[0] += -0.050033772213187305;
                      }
                    } else {
                      result[0] += 0.07713741216659205;
                    }
                  } else {
                    result[0] += -0.020506473770986373;
                  }
                } else {
                  if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)3.02604460716247603) ) ) {
                    result[0] += -0.02749452843132228;
                  } else {
                    result[0] += 0.15361405411813348;
                  }
                }
              } else {
                result[0] += -0.01689770900788975;
              }
            }
          } else {
            result[0] += -0.01980462952004321;
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.0035160400126013217;
        } else {
          result[0] += -0.11451476106617796;
        }
      } else {
        result[0] += 0.006497728788180464;
      }
    } else {
      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
          result[0] += -0.09014571285664397;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
            result[0] += 0.10851496053325296;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.11240996980053237;
            } else {
              result[0] += 0.022874333716064454;
            }
          }
        }
      } else {
        result[0] += 0.13966045012186407;
      }
    }
  } else {
    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
      result[0] += 0.00018521327309719903;
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
          result[0] += 0.1899587847622029;
        } else {
          result[0] += 0.030034717237519222;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.509355545043946201) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.08769498541856303;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.08476731612951363;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                          result[0] += 0.00961621043767718;
                        } else {
                          result[0] += -0.024397695587828956;
                        }
                      } else {
                        result[0] += 0.021165582252762594;
                      }
                    } else {
                      result[0] += -0.02336594601479635;
                    }
                  } else {
                    result[0] += 0.045055781364515796;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    result[0] += -0.12309700315681016;
                  } else {
                    result[0] += 0.0006597031354037357;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += -0.004845720607454061;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
                      result[0] += -0.03577352134657249;
                    } else {
                      result[0] += -0.0774333594944227;
                    }
                  }
                }
              } else {
                result[0] += 0.04144888195085747;
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.77930498123169123) ) ) {
                      result[0] += -0.0820433386013834;
                    } else {
                      result[0] += 0.04788974551065839;
                    }
                  } else {
                    result[0] += 0.06164576465498837;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.153024196624756748) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                          result[0] += -0.08899629370571895;
                        } else {
                          result[0] += 0.008535767197560669;
                        }
                      } else {
                        result[0] += -0.12338854682546069;
                      }
                    } else {
                      result[0] += 0.024172739686020656;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += 0.01467133466090081;
                    } else {
                      result[0] += -0.02503315560356072;
                    }
                  }
                }
              } else {
                result[0] += -0.005006932769438564;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.147398948669434482) ) ) {
                result[0] += -0.0007587851820579021;
              } else {
                result[0] += -0.018527727818732725;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.00843093865174559;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.05286061297711534;
                  } else {
                    result[0] += -0.10360871714442682;
                  }
                } else {
                  result[0] += -0.03187193614774605;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.313699722290040839) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.03523410535618595;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.1736095522884592;
                    } else {
                      result[0] += 0.04222598101716558;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.14287701805603042;
                  } else {
                    result[0] += 0.0010117816492299157;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += -0.021344888482397066;
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.04423801955145585;
                      } else {
                        result[0] += -0.028426611421574113;
                      }
                    } else {
                      result[0] += 0.11260406266435628;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.09811223704073793;
                    } else {
                      result[0] += -0.04828733831815521;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.338562726974488193) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.09415697355915788;
              } else {
                result[0] += -0.013452233620912461;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += -0.023409126691878987;
                } else {
                  result[0] += 0.07378001098299888;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.55753517150879084) ) ) {
                    result[0] += -0.001969154923198878;
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02450091362514052;
                    } else {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0115711969124153;
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
                          result[0] += 0.021945137930806463;
                        } else {
                          result[0] += 0.0831483202339967;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.04947813634423312;
                    } else {
                      result[0] += 0.00063156291057281;
                    }
                  } else {
                    result[0] += 0.15064914855218073;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.0033882517681382215;
        } else {
          result[0] += -0.11057208822916934;
        }
      } else {
        result[0] += 0.007051361762264343;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
        result[0] += 0.18857393935562441;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          result[0] += -0.12945414122195875;
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.795426130294800249) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)2.381086945533752885) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.1543984311747721;
                } else {
                  result[0] += 0.024847654117810678;
                }
              } else {
                result[0] += 0.10827698726304597;
              }
            } else {
              result[0] += -0.08335950141566437;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += -0.07335064694058961;
                  } else {
                    result[0] += 0.005428831934170323;
                  }
                } else {
                  result[0] += 0.10184569306324776;
                }
              } else {
                result[0] += 0.11082507220309759;
              }
            } else {
              result[0] += -0.08033957038762972;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
              result[0] += 0.01916854042138259;
            } else {
              result[0] += -0.03823373526416293;
            }
          } else {
            result[0] += -0.09591273236975931;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.018893271113693086;
            } else {
              result[0] += 0.027898572115290305;
            }
          } else {
            result[0] += -0.028683212047680703;
          }
        }
      } else {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.285735368728638583) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.0011500145084654013;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.029487809419706114;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.024594869722415914;
                  } else {
                    result[0] += 0.023816087073621938;
                  }
                } else {
                  result[0] += 0.005501697941896747;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.055496215820313388) ) ) {
              result[0] += 0.009513656200120233;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.005790161651609218;
              } else {
                result[0] += -0.008462169098962526;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.003486979787694362;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8195080757141131) ) ) {
                  result[0] += 0.04628993587268761;
                } else {
                  result[0] += 0.002688216420955722;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0014464157705769759;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.006912331188176086;
                  } else {
                    result[0] += -0.035211302677350696;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.001163536661474987;
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.44140100479126021) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.059795096939538855;
                } else {
                  result[0] += -0.014105029263709705;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.004073202424947726;
                      } else {
                        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.0026336971180493077;
                        } else {
                          result[0] += -0.025208995553354503;
                        }
                      }
                    } else {
                      result[0] += -0.043411981541039465;
                    }
                  } else {
                    result[0] += 0.046951552307704425;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.009970779734948271;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.07713842943228781;
                    } else {
                      result[0] += -0.000819760555868777;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
        result[0] += -0.0013401188014675075;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.0023855488763650933;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10889482498169123) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.591613531112671787) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += 0.05804056836299729;
                  } else {
                    result[0] += 0.012845626458707541;
                  }
                } else {
                  result[0] += -0.0060949344015424275;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += 0.007288010832528205;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.006200042755855579;
                  } else {
                    result[0] += -0.06574789269305463;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.011593746871568318;
                } else {
                  result[0] += 0.004389889299391338;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.005667706246818807;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += -0.010208873720781443;
                    } else {
                      result[0] += 0.05230650670234075;
                    }
                  }
                } else {
                  result[0] += 0.0006766227237457535;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0015373941492332296;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.009241586841493591;
                } else {
                  result[0] += -0.03064919183554119;
                }
              }
            } else {
              result[0] += 0.03345964154614247;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.0034501436539224233;
        } else {
          result[0] += -0.10997644893954983;
        }
      } else {
        result[0] += 0.005868725150478288;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.970085620880127397) ) ) {
        result[0] += 0.19305762495284107;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
          result[0] += 0.09224524959328924;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += -0.12095166272932828;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
              result[0] += -0.0786300741951612;
            } else {
              result[0] += 0.026072752138884943;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.952717304229737216) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
              result[0] += -0.0009441046105302694;
            } else {
              result[0] += 0.05849121370561372;
            }
          } else {
            result[0] += -0.01343259462157255;
          }
        } else {
          result[0] += -0.025725668231330247;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0068312427691172885;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                    result[0] += -0.03700207051160618;
                  } else {
                    result[0] += -0.004926902220501239;
                  }
                }
              } else {
                result[0] += -0.016615541668563768;
              }
            } else {
              result[0] += 0.005435782166657148;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
                result[0] += 0.08714437904002303;
              } else {
                result[0] += -0.007081108911112505;
              }
            } else {
              result[0] += -0.03207739114504992;
            }
          }
        } else {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -9.326785582175812e-05;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.173316955566407138) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.010856982391941092;
                } else {
                  result[0] += -0.010725727965096117;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.877672910690308505) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.0020766501447870545;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          result[0] += 0.09675052861364868;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                            result[0] += -0.12045042346545645;
                          } else {
                            result[0] += -0.004109049360513842;
                          }
                        }
                      } else {
                        result[0] += -0.03143362467092122;
                      }
                    }
                  } else {
                    result[0] += 0.005866750555313806;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.006570774855368725;
                      } else {
                        result[0] += -0.16243940602141585;
                      }
                    } else {
                      result[0] += 0.010715866305219594;
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += 0.02574320894256256;
                    } else {
                      result[0] += -0.04314245343046569;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.002710801717683013;
              } else {
                if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.00032870778588049395;
                  } else {
                    result[0] += 0.05552059783230491;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.04118632513563649;
                  } else {
                    result[0] += 0.013474382903721528;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        result[0] += 0.0008197492987363276;
      } else {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.06747679673379932;
                } else {
                  result[0] += 0.007803225655511273;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.0004693804305928345;
                } else {
                  result[0] += -0.02066368676692819;
                }
              }
            } else {
              result[0] += -0.037162064129402664;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.015826500828670022;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.02387032102885958;
              } else {
                result[0] += 0.004460821038499885;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.02999357201446718;
                    } else {
                      result[0] += -0.003484115686234071;
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.93885374069213956) ) ) {
                      result[0] += 3.695151456319757e-05;
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.03922955386461871;
                        } else {
                          result[0] += 0.03607213400808902;
                        }
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          result[0] += 0.003977718863856972;
                        } else {
                          result[0] += -0.01049319991547851;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0025203461158792226;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                      result[0] += -0.025306500503281856;
                    } else {
                      result[0] += 0.009355505020446206;
                    }
                  }
                }
              } else {
                result[0] += -0.01950254352381616;
              }
            } else {
              result[0] += 0.012568562897535128;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += -0.0009774781657594124;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.005474198051970015;
              } else {
                result[0] += 0.05872841029493259;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.002928102485627059;
        } else {
          result[0] += -0.10973695342664308;
        }
      } else {
        result[0] += 0.005581021408672263;
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.970085620880127397) ) ) {
        result[0] += 0.19359668216452186;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
          result[0] += 0.012280130561323429;
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
              result[0] += -0.00921644879688164;
            } else {
              result[0] += 0.112994457520033;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.809641838073731357) ) ) {
              result[0] += 0.11435454814880452;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                result[0] += -0.13627049227613194;
              } else {
                result[0] += 0.10763687670998814;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.29409265518188654) ) ) {
          result[0] += 0.000455036532054279;
        } else {
          result[0] += 0.00968790253201491;
        }
      } else {
        result[0] += -0.013291749477962828;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.002775086790104987;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                    result[0] += 0.016543222392652667;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.028480233237342657;
                      } else {
                        result[0] += 0.003184521321582909;
                      }
                    } else {
                      result[0] += 0.015299676626409901;
                    }
                  }
                } else {
                  result[0] += 0.026226068028701123;
                }
              }
            } else {
              result[0] += 0.0009236342448773158;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.010114088947811817;
              } else {
                result[0] += -0.0213278979363048;
              }
            } else {
              result[0] += -0.006003240064531037;
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += 0.01320064774079819;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.032269203625214286;
                      } else {
                        result[0] += 0.05577314085225143;
                      }
                    } else {
                      result[0] += 0.013768182235834155;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.029743213453101602;
                  } else {
                    result[0] += -0.0771537724755123;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01496048696821156;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.050107120584160864;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.0171258857530723;
                    } else {
                      result[0] += -0.02393643384219897;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                  result[0] += -0.0029326784755143416;
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += 0.05591563072445121;
                    } else {
                      result[0] += -0.061263946250170766;
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.014150807042208764;
                    } else {
                      result[0] += -0.043833307986239564;
                    }
                  }
                }
              } else {
                result[0] += 0.00087075355229087;
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                result[0] += -0.0238911134667502;
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.022915579345480063;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.041433153328790426;
                      } else {
                        result[0] += -0.027608717555321207;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                        result[0] += 0.008856646114198908;
                      } else {
                        result[0] += 0.04032956406979583;
                      }
                    }
                  } else {
                    result[0] += -0.023435616774349327;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.005138693832973543;
                } else {
                  result[0] += -0.029287084236529523;
                }
              } else {
                result[0] += 0.004470804057986752;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.00713991153142465;
          } else {
            result[0] += -0.00094238006816217;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
              result[0] += 0.021461395608967173;
            } else {
              result[0] += -0.006308766269173484;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.013648325891220656;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.88338184356689631) ) ) {
                  result[0] += 0.0024621999971733534;
                } else {
                  result[0] += -0.015615454310089746;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.0015966502833778679;
                          } else {
                            result[0] += -0.03503172419431603;
                          }
                        } else {
                          result[0] += -5.4716442602862336e-05;
                        }
                      } else {
                        result[0] += -0.03279402091407239;
                      }
                    } else {
                      result[0] += -0.03941693924519344;
                    }
                  } else {
                    result[0] += -0.004372357379775176;
                  }
                } else {
                  result[0] += -0.00027850360938753375;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
            result[0] += 0.05614907944769668;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                  result[0] += -0.0007665538287138704;
                } else {
                  result[0] += -0.024191934493864017;
                }
              } else {
                result[0] += 0.002446427329486026;
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.10005309093837039;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.05039608663805514;
                } else {
                  result[0] += -0.011255502071279719;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.014281765585243373;
          } else {
            result[0] += -0.06924818785745004;
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.1717605590693202;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.008794729050955762;
          } else {
            result[0] += -0.04416769189858189;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.000383682035941755;
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.478159427642823154) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.19325071060072885;
              } else {
                result[0] += 0.008366451661444901;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.011539532299868229;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
                    result[0] += 0.02068386662721331;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.0034754237399206656;
                    } else {
                      result[0] += -0.28263533740972135;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.985194206237793857) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.45749855041504084) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.509355545043946201) ) ) {
                            result[0] += -0.018131368082066276;
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.591613531112671787) ) ) {
                              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += 0.024465714213849948;
                              } else {
                                result[0] += -0.024823674093125987;
                              }
                            } else {
                              result[0] += -0.0299543819517017;
                            }
                          }
                        } else {
                          result[0] += 0.09974330577852564;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.04249214863676434;
                          } else {
                            result[0] += 0.1342575504174517;
                          }
                        } else {
                          result[0] += -0.08905217505076231;
                        }
                      }
                    } else {
                      result[0] += 0.020373694141362007;
                    }
                  } else {
                    result[0] += 0.005931114337413439;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.87162971496582209) ) ) {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.000000000000000888) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.0023927364780083793;
                          } else {
                            result[0] += -0.011729513829998732;
                          }
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += 0.012748080053402564;
                            } else {
                              result[0] += -0.046557880406590424;
                            }
                          } else {
                            result[0] += -0.007166455313407538;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                            result[0] += -0.03791224767187713;
                          } else {
                            result[0] += -0.0033801480147371728;
                          }
                        } else {
                          result[0] += -0.06103382500538623;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.03003196868225987;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                              result[0] += 0.10664561007121937;
                            } else {
                              result[0] += 0.010786208699248496;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                            result[0] += 0.10158716058041338;
                          } else {
                            result[0] += 0.02701533931872957;
                          }
                        }
                      } else {
                        result[0] += 0.09755190228516818;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.0002661282989144634;
                      } else {
                        result[0] += -0.04558803452007142;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.013013186301815248;
                          } else {
                            result[0] += -0.08516670824952595;
                          }
                        } else {
                          result[0] += 0.014451763718245095;
                        }
                      } else {
                        result[0] += 0.044990410507639626;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.013180544789411467;
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.005333996951802793;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.00981957491047125;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                result[0] += 0.011490956026989895;
              } else {
                result[0] += 0.05553488574657487;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += 0.012148987630369902;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.02628833408257939;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.10145450333713567;
              } else {
                result[0] += -0.043304140495549946;
              }
            }
          }
        } else {
          result[0] += -0.004349240923839964;
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.006027016742962379;
          } else {
            result[0] += -0.027285692141173285;
          }
        } else {
          result[0] += -0.002682817233966016;
        }
      }
    } else {
      result[0] += 1.386360938489265e-05;
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.12409635576905553;
        } else {
          result[0] += -0.0006608835314723581;
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.05681927771125981;
                  } else {
                    result[0] += 0.10601254458872855;
                  }
                } else {
                  result[0] += 0.12200575183936609;
                }
              } else {
                result[0] += -0.009940978437727455;
              }
            } else {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.19353129159900243;
              } else {
                result[0] += -0.034682328411597324;
              }
            }
          } else {
            result[0] += 0.02064884791652302;
          }
        } else {
          result[0] += 0.018605707531098095;
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.03542072966189106;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.76710987091064631) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.855921268463135654) ) ) {
                      result[0] += -0.01356299915765358;
                    } else {
                      result[0] += -0.14624838820998073;
                    }
                  } else {
                    result[0] += 0.036568165621041836;
                  }
                } else {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.08147136337447923;
                  } else {
                    result[0] += -0.01578986890906894;
                  }
                }
              } else {
                result[0] += -0.004441389103103728;
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.03432561251362719;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.00795052428874923;
              } else {
                result[0] += -0.16756854295814508;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.005814216995444067;
            } else {
              result[0] += -0.06699991782347736;
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.591613531112671787) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.561121463775635654) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.85118436813354581) ) ) {
                    result[0] += 0.0356975100045006;
                  } else {
                    result[0] += -0.2034788491569698;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                    result[0] += -0.16692607631345724;
                  } else {
                    result[0] += -0.02066284097131371;
                  }
                }
              } else {
                result[0] += 0.02844689802800601;
              }
            } else {
              result[0] += 0.04346252452680576;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.531673669815064365) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.012360113670547833;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                      result[0] += -0.0027309619977399363;
                    } else {
                      result[0] += 0.0156860171616619;
                    }
                  } else {
                    result[0] += -0.009664530074333065;
                  }
                }
              } else {
                result[0] += -0.02235261191012675;
              }
            } else {
              result[0] += -0.03455378889793058;
            }
          } else {
            result[0] += 0.1849521185703912;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.33581686019897639) ) ) {
                  result[0] += 0.02420151614346333;
                } else {
                  result[0] += 0.18454441591522333;
                }
              } else {
                result[0] += -0.03739074426252404;
              }
            } else {
              result[0] += -0.002083241374366267;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.572941064834595615) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.00019360373221087465;
                    } else {
                      result[0] += -0.02582276065275374;
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.003883330433700376;
                    } else {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.005633345884316252;
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.004601820496493001;
                        } else {
                          result[0] += 0.06976591648927337;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.022137695031782097;
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.0036556474271840743;
                    } else {
                      result[0] += -0.020974912999133823;
                    }
                  }
                }
              } else {
                result[0] += 0.012906310161171658;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0037737940546387246;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)13.86781549453735529) ) ) {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.011592420958267156;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.86783456802368342) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.413873195648194248) ) ) {
                              result[0] += -0.006658088129639305;
                            } else {
                              result[0] += -0.14423807165262287;
                            }
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += -0.10330662656868168;
                            } else {
                              result[0] += 0.011869455918637526;
                            }
                          }
                        } else {
                          result[0] += 0.013375339786713512;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.019815072237141282;
                        } else {
                          result[0] += 0.07403199335942724;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.25862455368042081) ) ) {
                        result[0] += 0.10381993915265371;
                      } else {
                        if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.07734509115744446;
                        } else {
                          result[0] += 0.010223508346540712;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.11773479304709521;
                }
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.0006364480674276684;
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.000000000000000888) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.0009268253619927758;
              } else {
                result[0] += -0.013669277970180888;
              }
            } else {
              result[0] += 0.00864324722310155;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.667095184326172763) ) ) {
              result[0] += 0.0020475566286354547;
            } else {
              result[0] += 0.05142577713374833;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.04635243085252717;
          } else {
            result[0] += -5.1761971342653454e-05;
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.10224588874431165;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.014889119414554073;
          } else {
            result[0] += -0.0775847088133923;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.855006217956543857) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
                  result[0] += 0.006563515883142615;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.008167218126655818;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.13229258904325242;
                      } else {
                        result[0] += -0.03459299870982267;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += -0.038142063630280505;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.006497240403575174;
                        } else {
                          if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.020537208985725783;
                          } else {
                            result[0] += 0.021798144194970122;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.418550252914429599) ) ) {
                        result[0] += -0.06214499700932985;
                      } else {
                        result[0] += -0.009733920491782027;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.019506394898123572;
                } else {
                  result[0] += -0.009495644969551941;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.005411579532618418;
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.022356916249223525;
                  } else {
                    result[0] += 0.06242717144917123;
                  }
                }
              } else {
                result[0] += 0.008098003109532757;
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.02683888172217295;
                  } else {
                    result[0] += 0.0015916845360304647;
                  }
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.01573707198915728;
                    } else {
                      result[0] += 0.006326748705075215;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += 0.0400656065973762;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.0036010620688833195;
                        } else {
                          result[0] += -0.03967963114775738;
                        }
                      } else {
                        result[0] += 0.045516563054445694;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.06825513165894193;
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.03146473030484983;
              } else {
                result[0] += -0.05778171588150756;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += -0.04062547997226508;
          } else {
            result[0] += -0.002547109231583547;
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += -0.004445316910225888;
          } else {
            result[0] += 0.18502202255865258;
          }
        } else {
          result[0] += 0.0014023362311201431;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.03273979509717572;
            } else {
              result[0] += 0.00777559848245904;
            }
          } else {
            result[0] += 0.0008748344192453765;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -8.260959333604208e-05;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.002468859347830789;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.021175786003644916;
                } else {
                  result[0] += 0.011941741459977194;
                }
              } else {
                result[0] += -0.03729410791650068;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.377587795257569248) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.02981950703739586;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.019963634037718062;
                  } else {
                    result[0] += 0.009344975223805571;
                  }
                }
              } else {
                result[0] += 0.050047518898806036;
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.01343888814226772;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += -0.0056018343439096455;
                } else {
                  result[0] += 0.06298246417791464;
                }
              }
            }
          } else {
            result[0] += -0.011360001847495487;
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.007177565782298967;
          } else {
            result[0] += -0.02271371050398254;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += 0.00029088102114346534;
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.147398948669434482) ) ) {
          result[0] += -0.004110180620175163;
        } else {
          result[0] += -0.02664331176932029;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.003838300704956943) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.113908529281617099) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.097527027130127841) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                      result[0] += -0.047852541584831654;
                    } else {
                      result[0] += 0.05569912952952877;
                    }
                  } else {
                    result[0] += -0.0072132602112717145;
                  }
                } else {
                  result[0] += -0.1809094728136182;
                }
              } else {
                result[0] += 0.005734736587226137;
              }
            } else {
              result[0] += 0.011692590884707199;
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.008113540332261067;
            } else {
              result[0] += -0.03818469380659232;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
            result[0] += 0.005531534197291389;
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.0054741281058514665;
            } else {
              result[0] += 0.01372281343905226;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.531673669815064365) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += 0.029126177066476834;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.03090598752520923;
                  } else {
                    result[0] += -0.010264463867797488;
                  }
                } else {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.15369039664037243;
                  } else {
                    result[0] += -0.034907603959697685;
                  }
                }
              } else {
                result[0] += -0.0016725322839391717;
              }
            }
          } else {
            result[0] += 0.1850886518967314;
          }
        } else {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.001337769538676357;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0009772681211515442;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.77930498123169123) ) ) {
                result[0] += -0.011669748317083995;
              } else {
                result[0] += 0.061086948642735464;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
          result[0] += -0.08070019950633114;
        } else {
          result[0] += -0.00463880980947577;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
          result[0] += 0.008311599802778206;
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.029931616381261284;
          } else {
            result[0] += 0.014350839212517404;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.293085813522339311) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.14695957179439756;
                    } else {
                      result[0] += 0.01849410080649905;
                    }
                  } else {
                    result[0] += 0.014804261306379827;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.293085813522339311) ) ) {
                    result[0] += -0.11041549010929637;
                  } else {
                    result[0] += -0.016973868372914765;
                  }
                }
              } else {
                result[0] += 0.02748648866670722;
              }
            } else {
              result[0] += -0.003219414331073421;
            }
          } else {
            result[0] += -0.02387848228384494;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
                result[0] += -0.02241794315615475;
              } else {
                result[0] += -0.09943719204889427;
              }
            } else {
              result[0] += 0.038665017235772994;
            }
          } else {
            result[0] += -0.0006136287341877602;
          }
        }
      } else {
        result[0] += -0.0004956483443005656;
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08277794248820045;
              } else {
                result[0] += 0.07325547662740525;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.07095210037916726;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.013797735480235893;
                } else {
                  result[0] += -0.12442686654564611;
                }
              }
            }
          } else {
            result[0] += -0.007539853670681883;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.00084278158628423;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
                      result[0] += 0.060636352363424406;
                    } else {
                      result[0] += -0.10953375176173281;
                    }
                  } else {
                    result[0] += 0.013278602933806748;
                  }
                } else {
                  result[0] += -0.021221212777107864;
                }
              }
            } else {
              result[0] += -0.009732985005227334;
            }
          } else {
            result[0] += 0.022748515843717163;
          }
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.764714598655701128) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.25333833694458185) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0012157004667741386;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                    result[0] += 0.0009345783170538211;
                  } else {
                    result[0] += -0.016163040407519227;
                  }
                }
              } else {
                result[0] += -0.01788146933203671;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.09253295060242073;
              } else {
                result[0] += -0.14678387881751098;
              }
            }
          } else {
            result[0] += -0.05491524516603322;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.02764630906145182;
            } else {
              result[0] += 0.009167943767625641;
            }
          } else {
            result[0] += -0.03439668232687812;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
              result[0] += -0.004254666723164078;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.13104699066241696;
                    } else {
                      result[0] += 0.10161731461276768;
                    }
                  } else {
                    result[0] += -0.0033701234336224297;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.024890166081046734;
                    } else {
                      result[0] += -0.009043613532887379;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                        result[0] += -0.05052690258813669;
                      } else {
                        result[0] += 0.02917231389857817;
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.02493802287956944;
                      } else {
                        result[0] += 0.06308895954706163;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  result[0] += 0.01739336479890274;
                } else {
                  result[0] += 0.0037626643291306136;
                }
              }
            }
          } else {
            result[0] += 0.0005804822301381667;
          }
        } else {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0008176510657893779;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += -0.00532944711200909;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.027703248375165745;
                } else {
                  result[0] += 0.023737183223169464;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += -0.10388849390897095;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += -0.024005030498097907;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
                    result[0] += 0.0713349230660653;
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0057952037193838854;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        result[0] += -0.03813245072956595;
                      } else {
                        result[0] += 0.06582629137709653;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.023521737855889;
                } else {
                  result[0] += -0.006812720366760793;
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.017351991404026438;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.0016270766749583443;
                  } else {
                    result[0] += 0.022982698543515662;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0738187837422173;
            } else {
              result[0] += -0.017915918776244266;
            }
          } else {
            result[0] += 0.004779313698753726;
          }
        } else {
          result[0] += 0.007591136917653012;
        }
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.006595813247798695;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += -0.03162341884189252;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.00991867279847885;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.005592293023083363;
                        } else {
                          result[0] += -0.05560899890771263;
                        }
                      } else {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.004863293306133607;
                        } else {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.007397465402339056;
                          } else {
                            result[0] += 0.04966522345906885;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.12409909802704847;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.02220938759140429;
                  } else {
                    result[0] += -0.04076358378419948;
                  }
                } else {
                  result[0] += 0.11013370873626083;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.992907285690308505) ) ) {
                result[0] += 0.009813666984226505;
              } else {
                result[0] += 0.07036579284553744;
              }
            }
          }
        } else {
          result[0] += -0.01578618114109303;
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
            result[0] += -0.03876254590952557;
          } else {
            result[0] += -0.15487692255382124;
          }
        } else {
          result[0] += 0.01709726861648927;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
        result[0] += -0.0005425727967228051;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
              result[0] += -0.010586925662072083;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.009381976948958887;
              } else {
                result[0] += 0.06452092487225912;
              }
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.059316430958539446;
              } else {
                result[0] += 0.008251998907538523;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.01881969853760599;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += -0.009569952887800624;
                  } else {
                    result[0] += 0.07370821392371464;
                  }
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.007139425693603513;
                } else {
                  result[0] += -0.04343280349447218;
                }
              }
            }
          }
        } else {
          result[0] += -0.015242428845114077;
        }
      }
    } else {
      result[0] += 4.0731988551768446e-05;
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.97393989562988459) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0018908495291083478;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.006535167265520252;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.08834172601353493;
                } else {
                  result[0] += 0.023680739142250138;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.026803855788442712;
                } else {
                  result[0] += -0.07742601117757549;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.028829340416532612;
                  } else {
                    result[0] += -0.05858317039448172;
                  }
                } else {
                  result[0] += 0.010298982181031981;
                }
              } else {
                result[0] += -0.004265815001925426;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                result[0] += 0.0019468962288634328;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.014258431512186859;
                } else {
                  result[0] += -0.03860143825339116;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                    result[0] += -0.0012638880957178905;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.075206041336061347) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.03352200360506014;
                          } else {
                            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.0027783877818058646;
                            } else {
                              result[0] += -0.06660008044101806;
                            }
                          }
                        } else {
                          result[0] += 0.00883642251449157;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.004769296485519176;
                          } else {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += -0.005729281226744973;
                                } else {
                                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                      result[0] += 0.009145893401232737;
                                    } else {
                                      result[0] += 0.03980208052426641;
                                    }
                                  } else {
                                    result[0] += 0.06275383216259049;
                                  }
                                }
                              } else {
                                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
                                  result[0] += -0.02911874715749719;
                                } else {
                                  result[0] += 0.10950944682280378;
                                }
                              }
                            } else {
                              result[0] += -0.007750861385144025;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += -7.970390279895151e-05;
                              } else {
                                result[0] += -0.1256829456610197;
                              }
                            } else {
                              result[0] += 0.016040430680956376;
                            }
                          } else {
                            result[0] += -0.008842950977335024;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.05962565537767675;
                    }
                  }
                } else {
                  result[0] += 0.1094404380518384;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.12123861359121642;
                    } else {
                      result[0] += 0.0019117736642898777;
                    }
                  } else {
                    result[0] += -0.030229794251999988;
                  }
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.003276847851861742;
                  } else {
                    result[0] += 0.014166511908602892;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.50380659103393732) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.042640521386591884;
                      } else {
                        result[0] += -0.0043300686992750696;
                      }
                    } else {
                      result[0] += -0.05221507997027419;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.552201986312867099) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.283562898635865146) ) ) {
                        result[0] += 0.0021331907259030036;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                          result[0] += -0.013762960209584252;
                        } else {
                          result[0] += 0.052990157272662;
                        }
                      }
                    } else {
                      result[0] += 0.02476695805825673;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                      result[0] += -0.08395647098857709;
                    } else {
                      result[0] += 0.01109528305764286;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.03420138359069913) ) ) {
                      result[0] += -0.078957812039457;
                    } else {
                      result[0] += 0.0061381203715202524;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.016899031963286053;
                } else {
                  result[0] += 0.04097281860996702;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.435900688171387607) ) ) {
            result[0] += -0.0006317672978051004;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
              result[0] += -0.018882013064813172;
            } else {
              result[0] += -0.16715296910268307;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += -0.06644264885588314;
            } else {
              result[0] += 0.05075724052004215;
            }
          } else {
            result[0] += -0.008894375548948103;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.019082374728316122;
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.342454433441162998) ) ) {
          result[0] += -0.06374349133118061;
        } else {
          result[0] += 0.029031516019900866;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
      result[0] += -0.0001157071489505104;
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.006624084849495862;
        } else {
          result[0] += 0.00032167938619466344;
        }
      } else {
        result[0] += -0.014343917516403865;
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.77930498123169123) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.10114623303399028;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.008492638407309763;
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.032005591422600684;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.6739082336425799) ) ) {
                            result[0] += -0.008065415018847914;
                          } else {
                            result[0] += 0.018007694513280815;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.0005452031844967757;
                        } else {
                          result[0] += 0.026383103482942535;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.04086866695588344;
                      } else {
                        result[0] += -0.09677568842016981;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0599189788590221;
                    } else {
                      result[0] += -0.018382503044380532;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.008349108320783859;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.085941076278687412) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                          result[0] += 0.002097549262635412;
                        } else {
                          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
                                result[0] += 0.011363782291679812;
                              } else {
                                result[0] += -0.05661503173436986;
                              }
                            } else {
                              result[0] += 0.016265869704183023;
                            }
                          } else {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.44140100479126021) ) ) {
                              result[0] += 0.002631766714641362;
                            } else {
                              result[0] += 0.049983326540242345;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.014339140601088355;
                            } else {
                              result[0] += 0.033718706129637566;
                            }
                          } else {
                            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.009260037259708394;
                            } else {
                              result[0] += -0.20825681821383688;
                            }
                          }
                        } else {
                          result[0] += 0.022079147107800186;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0017643469881870142;
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.47223544120788663) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                    result[0] += -0.0021757419798337503;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81443405151367365) ) ) {
                      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.06302418007781324;
                      } else {
                        result[0] += 0.0032971385938838312;
                      }
                    } else {
                      result[0] += 0.030165952110360023;
                    }
                  }
                } else {
                  result[0] += 0.0545329647223459;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.007580291799284382;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                        result[0] += 0.06357447306520353;
                      } else {
                        result[0] += -0.028306754952093085;
                      }
                    } else {
                      result[0] += -0.11913059411423532;
                    }
                  } else {
                    result[0] += -0.06363910309220965;
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.127005888433512;
                } else {
                  result[0] += 0.009621247827910864;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
                  result[0] += -0.054495034899724085;
                } else {
                  result[0] += 0.049811352233883235;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += 0.03587238549154901;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.547126770019532138) ) ) {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.283562898635865146) ) ) {
                            result[0] += -0.028913451643320626;
                          } else {
                            result[0] += 0.10855017595660817;
                          }
                        } else {
                          result[0] += 0.05870807480791321;
                        }
                      } else {
                        result[0] += 0.059683252992921466;
                      }
                    } else {
                      result[0] += -0.033677984552914306;
                    }
                  }
                } else {
                  result[0] += 0.0510547589827559;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0025675688273327957;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                result[0] += 0.00952662065693252;
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.021436171886731085;
                } else {
                  result[0] += 0.058912240261800836;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
              result[0] += 0.03355232843265208;
            } else {
              result[0] += -0.056429325249239506;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.435900688171387607) ) ) {
            result[0] += -0.002237735563424882;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
              result[0] += -0.018217322633336804;
            } else {
              result[0] += -0.17739597192343048;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.05926050359125067;
            } else {
              result[0] += 0.030016785733230595;
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.018452438422036137;
            } else {
              result[0] += 0.029567480885665926;
            }
          }
        }
      }
    } else {
      result[0] += 0.013264615867553814;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.24049568176269709) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -2.532316899242766e-05;
      } else {
        result[0] += -0.011404658548810955;
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.0024768474280534233;
      } else {
        result[0] += -0.01362127514082495;
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
      result[0] += 0.0739016010923891;
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.644374847412110263) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.16532567572824314;
                } else {
                  result[0] += -0.0014751385209986075;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.03614614521009743;
                } else {
                  result[0] += 0.12157001195353007;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.05408049062997992;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += -0.0600572048830236;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                        result[0] += 0.08995516928165732;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                          result[0] += 0.06670552690962385;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                            result[0] += -0.07443559627498303;
                          } else {
                            result[0] += 0.06455176650638304;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                            result[0] += 0.10646709153245;
                          } else {
                            result[0] += -0.042241081953609934;
                          }
                        } else {
                          result[0] += -0.058741063685065945;
                        }
                      } else {
                        result[0] += 0.0024793699556021386;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.241300821304322177) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                      result[0] += 0.04420402902654659;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                        result[0] += -0.16490593389116415;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.07531430676423408;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                            result[0] += -0.10494842006857341;
                          } else {
                            result[0] += 0.06416175799085727;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.350240230560303178) ) ) {
                      result[0] += 0.029365299458496363;
                    } else {
                      result[0] += -0.11908869804292335;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.09726025496842071;
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.684611082077027255) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.07369947676789448;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.04997460887020126;
                  } else {
                    result[0] += 0.06030615621680516;
                  }
                } else {
                  result[0] += -0.08645179612593934;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.743881702423096591) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.1014666187473061;
                    } else {
                      result[0] += 0.0028527068360116046;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.731793165206910068) ) ) {
                      result[0] += 0.06991338222630238;
                    } else {
                      result[0] += -0.0191973415821967;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.023707389831544745) ) ) {
                    result[0] += 0.06950399176849477;
                  } else {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.020573508124630133;
                    } else {
                      result[0] += 0.05560361687722144;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                  result[0] += 0.13903872877639908;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.847591876983644354) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.431880712509156162) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.03402869358119908;
                        } else {
                          result[0] += 0.10526068701502633;
                        }
                      } else {
                        result[0] += 0.11345341024146086;
                      }
                    } else {
                      result[0] += -0.06420431806243741;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += 0.14381469427831695;
                    } else {
                      result[0] += -0.0378823631814236;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.09462317691725297;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                result[0] += 0.09237295319590144;
              } else {
                result[0] += -0.05113203785337076;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.1003513398742691;
            } else {
              result[0] += -0.03300027110257881;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.07121736185692885;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                result[0] += 0.08588667883409386;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += 0.09668919021245875;
                } else {
                  result[0] += -0.03619065809305098;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
              result[0] += 0.13387704606305204;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.297764539718628818) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                  result[0] += 0.08080310303683534;
                } else {
                  result[0] += 0.006604158220477183;
                }
              } else {
                result[0] += -0.06533977562745989;
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.85118436813354581) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.12897505480356095;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.658652544021607333) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.898525714874268466) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.611299991607666904) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51013088226318537) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.749261140823365146) ) ) {
                          result[0] += -0.05052712092772721;
                        } else {
                          result[0] += 0.025146481782247484;
                        }
                      } else {
                        result[0] += -0.10069177897978794;
                      }
                    } else {
                      result[0] += -0.16550553346788627;
                    }
                  } else {
                    result[0] += 0.03325036466977561;
                  }
                } else {
                  result[0] += -0.13883120141882874;
                }
              }
            } else {
              result[0] += 0.041999843477984806;
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.00032034795912240093;
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.007285796297605302;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.08004762002823877;
          } else {
            result[0] += -0.013619169096688847;
          }
        }
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.004991323266010068;
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.00755735915067276;
          } else {
            result[0] += 0.07321129526434543;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
        result[0] += 0.11140370471612973;
      } else {
        result[0] += 0.01928763889000313;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.1270360946655309) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += -0.00028248568263955514;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
            result[0] += 0.020589840313862934;
          } else {
            result[0] += -0.07027439046697001;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.324458837509156162) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.006686651820021572;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.247078418731690341) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.594915628433228427) ) ) {
                                result[0] += -0.005967820874680079;
                              } else {
                                result[0] += 0.04007949909254512;
                              }
                            } else {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                                result[0] += 0.18874546930212072;
                              } else {
                                result[0] += 0.04478241382041449;
                              }
                            }
                          } else {
                            result[0] += -0.012209408458286101;
                          }
                        }
                      } else {
                        result[0] += -0.061973512272273545;
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                          result[0] += 0.03352433531614548;
                        } else {
                          result[0] += -0.020670034050474545;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                          result[0] += -0.017146348861214605;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                                result[0] += 0.07490277951387546;
                              } else {
                                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.825982809066773349) ) ) {
                                  result[0] += 0.006716726895655923;
                                } else {
                                  result[0] += 0.1486891835919323;
                                }
                              }
                            } else {
                              result[0] += -0.0011283339078043916;
                            }
                          } else {
                            result[0] += 0.12242020813761263;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.05173097301908212;
                  }
                } else {
                  result[0] += 0.06788019604790382;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.543910741806031162) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.0671655041763042;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.06730871509002093;
                    } else {
                      result[0] += 0.14054541928426095;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.02157529811737234;
                      } else {
                        result[0] += -0.05675654438060567;
                      }
                    } else {
                      result[0] += -0.043810154241496584;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                        result[0] += 0.1025175502060832;
                      } else {
                        result[0] += -0.04137081358308419;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += 0.16221828328220111;
                      } else {
                        result[0] += -0.021306061369252317;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.04214184939337128;
            }
          } else {
            result[0] += -0.07227677164390561;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.022708945988335377;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.93259048461914418) ) ) {
                result[0] += 0.004910203778503622;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.08963098614249605;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.08096114722756109;
                  } else {
                    result[0] += -0.0001005602828328256;
                  }
                }
              }
            }
          } else {
            result[0] += 0.1342051829925399;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                  result[0] += 0.08210562074728031;
                } else {
                  result[0] += 0.003197326402289462;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.048248677915940834;
                    } else {
                      result[0] += -0.03133751086741935;
                    }
                  } else {
                    result[0] += -0.05758707304976214;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += -0.09338048759866849;
                  } else {
                    result[0] += 0.028002316444762445;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.736135363578796831) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                  result[0] += 0.1759784149553112;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.14480076619948476;
                  } else {
                    result[0] += 0.01727664403998427;
                  }
                }
              } else {
                result[0] += 0.017763073051531738;
              }
            }
          } else {
            result[0] += -0.18045122846332914;
          }
        }
      } else {
        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.01719028097616756;
            } else {
              result[0] += -0.0831041156084728;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.031640254148035084;
            } else {
              result[0] += -0.05465711150070243;
            }
          }
        } else {
          result[0] += -0.04010969209088914;
        }
      }
    }
  }
  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.1270360946655309) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.01754331411910906;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += 0.14070176771283624;
                } else {
                  result[0] += 0.0033391914046191117;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                result[0] += 0.03338517905302252;
              } else {
                result[0] += 0.1237542157758151;
              }
            }
          } else {
            result[0] += -0.00142537187919934;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11978769302368342) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
              result[0] += 0.07577183259216547;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                result[0] += 0.07987322221535378;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                      result[0] += -0.06706428472131704;
                    } else {
                      result[0] += 0.06475306594680318;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.013381614975592475;
                    } else {
                      result[0] += -0.13791253834481348;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920601367950440341) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.725620865821838823) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += 0.015091202002882238;
                        } else {
                          result[0] += -0.10221218160433797;
                        }
                      } else {
                        result[0] += 0.05657007933760208;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.817651987075806108) ) ) {
                        result[0] += -0.09699745227117691;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.978102684020996982) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                            result[0] += 0.11446742189944276;
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.658699750900269443) ) ) {
                              result[0] += 0.0718394745177701;
                            } else {
                              result[0] += -0.10469275302977595;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.610145330429078037) ) ) {
                            result[0] += -0.09300287878989028;
                          } else {
                            result[0] += 0.01259880427270539;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                      result[0] += -0.13656960082719596;
                    } else {
                      result[0] += 0.017179235142229767;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                result[0] += 0.12733935722817738;
              } else {
                result[0] += 0.0214360530067442;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
                result[0] += 0.15970589707670146;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.457529306411744052) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.733257532119751865) ) ) {
                    result[0] += 0.09127636583915877;
                  } else {
                    result[0] += -0.02967624260705859;
                  }
                } else {
                  result[0] += 0.18445586579458095;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.842092990875245029) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.611299991607666904) ) ) {
              result[0] += -0.03953166758540372;
            } else {
              result[0] += -0.14703089783143947;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              result[0] += 0.04998754331101145;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.574378013610840732) ) ) {
                result[0] += 0.0024866522910342026;
              } else {
                result[0] += -0.19024438849844746;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.1172598444460069;
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06346921592855347;
                } else {
                  result[0] += 0.02318850623142218;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.039582058782505036;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                  result[0] += 0.08775792814076695;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                    result[0] += 0.06905384603889746;
                  } else {
                    result[0] += -0.03801107001410389;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
              result[0] += 0.08700613410685878;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.09321408896314369;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.297764539718628818) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.009397198783720931;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.08621758338399906;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.436733961105347568) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                            result[0] += 0.01949511618818761;
                          } else {
                            result[0] += 0.1114796793597178;
                          }
                        } else {
                          result[0] += -0.0826083458167984;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.59565925598144709) ) ) {
                          result[0] += 0.017505236085315495;
                        } else {
                          result[0] += 0.1434600094702075;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                    result[0] += -0.08959729266203387;
                  } else {
                    result[0] += -0.0014879887745121152;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.00035745544977665024;
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.12493765765588019;
        } else {
          result[0] += 0.008575413621421281;
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.69204592704773038) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.13592191372007065;
              } else {
                result[0] += -0.027838610418226224;
              }
            } else {
              result[0] += -0.002408432765694579;
            }
          } else {
            result[0] += -0.15676442648892513;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.797575235366822177) ) ) {
            result[0] += 0.17084104127664718;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.04930808843651016;
            } else {
              result[0] += -0.03687972534480313;
            }
          }
        }
      }
    } else {
      result[0] += -0.016846703462971468;
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.0004545665202941596;
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.407877445220948154) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.0008776109102672833;
                } else {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0153767475570392;
                    } else {
                      result[0] += -0.044589128021806285;
                    }
                  } else {
                    result[0] += 0.034530689860694234;
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.084203958511353427) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.013141266450299541;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.08025270065037551;
                            } else {
                              result[0] += -0.014048256471648691;
                            }
                          } else {
                            result[0] += 0.01076822221567091;
                          }
                        } else {
                          result[0] += -0.0026207078847708025;
                        }
                      } else {
                        result[0] += 0.007678110487712677;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.344720840454102451) ) ) {
                        result[0] += 0.03879380724988998;
                      } else {
                        result[0] += -0.019177747106821717;
                      }
                    } else {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.022116354514019627;
                        } else {
                          result[0] += -0.018923848351423217;
                        }
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.03866738454736586;
                        } else {
                          result[0] += -0.003399021404765955;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.007775315720260652;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01781293022989998;
                    } else {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.06129049981683754;
                      } else {
                        result[0] += 0.30371035465914126;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.011642180970514384;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.02134206468080113;
                  } else {
                    result[0] += 0.0026923651229054527;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.05033745271500581;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.04902015456505466;
                  } else {
                    result[0] += 0.006004619515265718;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.002623692615895786;
                  } else {
                    result[0] += -0.037641754656419266;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.06438822848295621;
                  } else {
                    result[0] += 0.008248384633663697;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0017381638575142023;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.027445147300107914;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.03339315585334955;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.046217358644408235;
                        } else {
                          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.023406927691204584;
                          } else {
                            result[0] += 0.13679814044853353;
                          }
                        }
                      } else {
                        result[0] += -0.037501984632335124;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += -0.09823421640193171;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.628555774688722479) ) ) {
                      result[0] += -0.017246058139078504;
                    } else {
                      result[0] += -0.003317910915378078;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.0026154095237314627;
                    } else {
                      result[0] += 0.09356450943700391;
                    }
                  } else {
                    result[0] += 0.0016322525379000323;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.001915239196193547;
                  } else {
                    result[0] += 0.02715725478947245;
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.039895389240444695;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.01259328178125465;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                            result[0] += -0.048295347634529286;
                          } else {
                            result[0] += 0.11624528405210002;
                          }
                        } else {
                          result[0] += 0.1202659521495338;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                            result[0] += -0.022570280511384253;
                          } else {
                            result[0] += 0.03249254495121852;
                          }
                        } else {
                          result[0] += 0.014436898023586615;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.02952402401549177;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.10448609099830455;
              } else {
                result[0] += 0.02058016531910733;
              }
            }
          } else {
            result[0] += 0.056119924334919996;
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += 0.018507231191933333;
          } else {
            result[0] += -0.009438651781527571;
          }
        } else {
          result[0] += -0.0338116145391783;
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.003104420928850599;
      } else {
        result[0] += -0.02087224199824336;
      }
    }
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.561121463775635654) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0855995925236824;
              } else {
                result[0] += -0.16333336789553646;
              }
            } else {
              result[0] += -0.0005195285104905085;
            }
          } else {
            result[0] += 0.011218459776108945;
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                result[0] += -0.03087866195061509;
              } else {
                result[0] += 0.020519311458903455;
              }
            } else {
              result[0] += 0.044348856886791396;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.003525959811582739;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.11160649349015395;
                  } else {
                    result[0] += -0.02068636440125419;
                  }
                }
              } else {
                result[0] += -0.05282784513387089;
              }
            } else {
              result[0] += -0.02725004454312507;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)13.86781549453735529) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += -0.028615336402816067;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.532256603240968573) ) ) {
                        result[0] += 0.018566776615396643;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.87162971496582209) ) ) {
                          result[0] += -0.11657802909207014;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.92430353164673029) ) ) {
                            result[0] += 0.11617790076033399;
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                              result[0] += 0.03148910207284996;
                            } else {
                              result[0] += -0.130067472786035;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.11250395479398895;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05479049682617365) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                            result[0] += -0.003492022012095457;
                          } else {
                            result[0] += 0.09450443337030463;
                          }
                        } else {
                          result[0] += -0.0424355207136286;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += -0.08554760653479009;
                    } else {
                      result[0] += -0.009310429418866197;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.659039497375490058) ) ) {
                    result[0] += -0.009046012468312459;
                  } else {
                    result[0] += 0.029972067004574984;
                  }
                }
              }
            } else {
              result[0] += -0.003794168469332869;
            }
          } else {
            result[0] += 0.12421544456567304;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.00025120739270532093;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                  result[0] += 0.027997004697321318;
                } else {
                  result[0] += 0.12204183772327898;
                }
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.381086945533752885) ) ) {
                    result[0] += -0.0449082605051025;
                  } else {
                    result[0] += 0.01270707186387554;
                  }
                } else {
                  result[0] += 0.03399120292627119;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.487163543701172763) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.00815594429547435;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                      result[0] += 0.00836719460157843;
                    } else {
                      result[0] += -0.012403909736989427;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.005257254468487127;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.003497707288420439;
                        } else {
                          result[0] += -0.09096965614459351;
                        }
                      }
                    } else {
                      result[0] += 0.03217988729854727;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.06438996304948172;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18980646133423029) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.013300452337198336;
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.022663895248320298;
                            } else {
                              result[0] += -0.10348691064031862;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                              result[0] += -0.05695728176563419;
                            } else {
                              result[0] += 0.056843076806915485;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.0413592839100115;
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.05447788007813018;
                          } else {
                            result[0] += 0.0363551859228885;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.068990230560303623) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.034801445309273725;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                          result[0] += -0.15704709524840665;
                        } else {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.10948538119051086;
                          } else {
                            result[0] += -0.04773073671271737;
                          }
                        }
                      } else {
                        result[0] += -0.007448235699546519;
                      }
                    }
                  } else {
                    result[0] += 0.012306570480282561;
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02121450088330767;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.021776013501982716;
                    } else {
                      result[0] += 0.03937141948268916;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.08976822827091957;
              } else {
                result[0] += 0.03145434233699794;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.007785317748768919;
    }
  } else {
    result[0] += -0.00040843637004284313;
  }
  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.003694815583722909;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
            result[0] += -0.0019208471383226672;
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.992712974548340732) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.03666209592056923;
              } else {
                result[0] += -0.005986807880908725;
              }
            } else {
              result[0] += -0.0010916332850406695;
            }
          }
        }
      } else {
        result[0] += 0.0008930193754037683;
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
        result[0] += -0.006246310966996783;
      } else {
        result[0] += -0.053803727439920734;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.29409265518188654) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.388278961181641513) ) ) {
                    result[0] += -0.006839393818531986;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.948021411895752841) ) ) {
                      result[0] += 0.054570006766153034;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                        result[0] += 0.5792648842475113;
                      } else {
                        result[0] += 0.09992672040012954;
                      }
                    }
                  }
                } else {
                  result[0] += 0.003183367176922513;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += 0.036951994118238206;
                  } else {
                    result[0] += 0.008838999803158797;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
                    result[0] += -0.0002259825761707606;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.04098080725780642;
                      } else {
                        result[0] += -0.04621978435411482;
                      }
                    } else {
                      result[0] += 0.030649426818318865;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.0029894610267190303;
              } else {
                result[0] += 0.03532089172786999;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.005318930486362708;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.057345739152160104;
                } else {
                  result[0] += 0.03977373187114866;
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.010538455139051248;
                  } else {
                    result[0] += -0.03324707286839192;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.012754477610055921;
                      } else {
                        result[0] += -0.010820324608602178;
                      }
                    } else {
                      result[0] += -0.01853356695438215;
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                        result[0] += 0.0024600938060645667;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.07563775599453192;
                            } else {
                              result[0] += 0.07100038402214193;
                            }
                          } else {
                            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += 0.0011591372382865083;
                            } else {
                              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.09482929912737748;
                              } else {
                                result[0] += -0.07835071818801614;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.049681106963322424;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                          result[0] += 0.03486119156894583;
                        } else {
                          result[0] += -0.01886807653182825;
                        }
                      } else {
                        if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.010350565644344544;
                        } else {
                          result[0] += -0.05411393928890651;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.06601969186741148;
                  } else {
                    result[0] += -0.008675670613150634;
                  }
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                          result[0] += -0.057994826409687394;
                        } else {
                          result[0] += 0.001807980311994525;
                        }
                      } else {
                        result[0] += -0.03303048470870031;
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                            result[0] += -0.15853762205680325;
                          } else {
                            result[0] += -0.022940441829981242;
                          }
                        } else {
                          result[0] += 0.005349278496538464;
                        }
                      } else {
                        result[0] += 0.02782593237923447;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.01706835204435878;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.023961207037817168;
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                              result[0] += -0.026729753850885992;
                            } else {
                              result[0] += 0.18146829166109602;
                            }
                          } else {
                            result[0] += 0.10339987238476295;
                          }
                        } else {
                          result[0] += 0.006792401527655297;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.026548230894895705;
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.08975387287797676;
              } else {
                result[0] += 0.019928786875720365;
              }
            }
          } else {
            result[0] += 0.047173639757268876;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += 0.01679327371898815;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.008794845788086742;
          } else {
            result[0] += -0.033826239844038014;
          }
        }
      }
    } else {
      result[0] += -0.01949691943873484;
    }
  }
  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)12.00000000000000178) ) ) {
    result[0] += 0.0003009673180408286;
  } else {
    if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.0007194369559092123;
      } else {
        result[0] += -0.027172673343257504;
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0031406886750985005;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                      result[0] += 0.016335841485371354;
                    } else {
                      result[0] += -0.013617569555605356;
                    }
                  } else {
                    result[0] += 0.016906310429364326;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.003774438433525675;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.579273939132691318) ) ) {
                        result[0] += -0.01618475039738836;
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                          result[0] += 0.02320751841268133;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.017797946929933417) ) ) {
                            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)6.000000000000000888) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                                result[0] += 0.8245572302704924;
                              } else {
                                result[0] += 4.064723449290562;
                              }
                            } else {
                              result[0] += 0.6372384090009441;
                            }
                          } else {
                            result[0] += 0.07553237306204102;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                            result[0] += -0.03899812773161041;
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                              result[0] += -0.04637043872506711;
                            } else {
                              result[0] += 0.33135306623500593;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
                            result[0] += 0.04725258732200599;
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.07786152071919561;
                            } else {
                              result[0] += 0.15428749202835818;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.04111465010067572;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.83939445018768355) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
                        result[0] += -0.0052968013573748566;
                      } else {
                        result[0] += -0.04877870043440768;
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                                result[0] += 0.0036757772641672903;
                              } else {
                                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                                  result[0] += -0.07284028686829067;
                                } else {
                                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                                    result[0] += -0.047733395362657347;
                                  } else {
                                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
                                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                                        result[0] += -0.10076912653603261;
                                      } else {
                                        result[0] += 1.0290485101773517;
                                      }
                                    } else {
                                      result[0] += 0.13156476242234053;
                                    }
                                  }
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                                result[0] += -0.06924222061866041;
                              } else {
                                result[0] += -0.0038355462145575506;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.297764539718628818) ) ) {
                                result[0] += 0.03722960502185279;
                              } else {
                                result[0] += -0.005662222363909461;
                              }
                            } else {
                              result[0] += -0.03663482099304631;
                            }
                          }
                        } else {
                          result[0] += -0.08306689503248438;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                            result[0] += 0.018354523532539753;
                          } else {
                            result[0] += 0.0868619378152378;
                          }
                        } else {
                          result[0] += -0.012728766526361946;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.042476749234206655;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                result[0] += 0.03555713878670011;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.002489176468307403;
                } else {
                  result[0] += -0.034659881742075536;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.914472818374634233) ) ) {
                    result[0] += 0.06468836543377517;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += 0.00391270642018751;
                        } else {
                          result[0] += 0.03958369653212876;
                        }
                      } else {
                        result[0] += -0.00648808222664872;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.827801465988160068) ) ) {
                        result[0] += 0.0531506660059382;
                      } else {
                        result[0] += 0.019454412338441612;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += 0.007949706684190247;
                  } else {
                    result[0] += -0.018658595502066076;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                    result[0] += 0.02657417059950548;
                  } else {
                    result[0] += -0.013653479947867468;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
                      result[0] += 0.024540443634614755;
                    } else {
                      result[0] += 0.21747535371043936;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                      result[0] += -0.011640144388021995;
                    } else {
                      result[0] += 0.07409205955046741;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.042999456493389265;
                } else {
                  result[0] += 0.04427125832692522;
                }
              } else {
                result[0] += -0.016474465415953682;
              }
            } else {
              result[0] += 0.051566180473151285;
            }
          } else {
            if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.007178517956406816;
            } else {
              result[0] += -0.02950666023408187;
            }
          }
        }
      } else {
        result[0] += -0.03246023509238901;
      }
    }
  }
}

