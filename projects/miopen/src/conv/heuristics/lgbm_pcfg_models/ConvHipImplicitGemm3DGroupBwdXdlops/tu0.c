
#include "header.h"

void predict_unit0(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.418141007423401323) ) ) {
      result[0] += 0.1612528710241388;
    } else {
      result[0] += 0.1951259687879441;
    }
  } else {
    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.020722113419978788;
          } else {
            result[0] += -0.1941699464793639;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.081657409667969194) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.14270338922828962;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.25930547714233576) ) ) {
                result[0] += 0.05844924771360023;
              } else {
                result[0] += -0.06847524715044098;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.11720784403338123;
            } else {
              result[0] += -0.19757089342748302;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.97661352157592951) ) ) {
                result[0] += -0.1212757773319164;
              } else {
                result[0] += 0.09175030746873813;
              }
            } else {
              result[0] += 0.14348935451583295;
            }
          } else {
            result[0] += -0.17967669933300529;
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.13879116054978713;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.799065828323365146) ) ) {
                  result[0] += 0.1627984198985891;
                } else {
                  result[0] += 0.026607021391020166;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.484580039978028232) ) ) {
                result[0] += -0.1420312204731217;
              } else {
                result[0] += -0.1923375782142294;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.15253155108386005;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.48918962478637873) ) ) {
                    result[0] += -0.15524212449115388;
                  } else {
                    result[0] += 0.14249330162933685;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.11739664790944873;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.04548945586507269;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.284418344497681552) ) ) {
                        result[0] += -0.08199720653627283;
                      } else {
                        result[0] += 0.06893827978533779;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.007886439984467029;
              } else {
                result[0] += 0.15315777614296985;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.18617508711397562;
        } else {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
              result[0] += -0.06491663662323088;
            } else {
              result[0] += -0.16141836859716233;
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.069195422530963;
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.07252597796955694;
              } else {
                result[0] += 0.005358175570726705;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
              result[0] += 0.15241461250658528;
            } else {
              result[0] += 0.014262555315391582;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.14636564802025573;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.10820708530088025;
                } else {
                  result[0] += -0.17880135821807036;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                result[0] += 0.1572768927232174;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.05374922005707276;
                } else {
                  result[0] += -0.12073097298341612;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.09886308707018975;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.13113981631119473;
                  } else {
                    result[0] += 0.023394643314023647;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                  result[0] += -0.10926988990624038;
                } else {
                  result[0] += 0.16522639578373619;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.284418344497681552) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.15505805402711315;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.16594327500566466;
                    } else {
                      result[0] += -0.03312482943726345;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.445705175399781162) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.17405570774636298;
                      } else {
                        result[0] += -0.021777145962055596;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.07480977502750343;
                      } else {
                        result[0] += 0.056376019277110424;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.13641265740541406;
                        } else {
                          result[0] += -0.036966000436410365;
                        }
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.07366390820303687;
                        } else {
                          result[0] += 0.18537369038977342;
                        }
                      }
                    } else {
                      result[0] += 0.19159066485847764;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.1848844489238659;
                } else {
                  result[0] += 0.027905125273810744;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.15431342662723757;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.694163918495178667) ) ) {
                result[0] += -0.007679797226119814;
              } else {
                result[0] += 0.19404330750466137;
              }
            }
          }
        }
      }
    }
  }
}

