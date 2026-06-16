
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.024733789176339407;
        } else {
          result[0] += 0.0668433521768058;
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += -0.003710062879245467;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY(  (data[28].missing != -1) && (data[28].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.10889307535950317;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.051834832186812334;
                } else {
                  result[0] += -0.031306954264837664;
                }
              }
            } else {
              result[0] += 0.05008102708300243;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
            result[0] += 0.02286255659665232;
          } else {
            result[0] += -0.020133869825552847;
          }
        }
      }
    } else {
      result[0] += 0.000456728302840944;
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.012675821781158891) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                    result[0] += 0.009590689215094296;
                  } else {
                    result[0] += -0.08261751023546422;
                  }
                } else {
                  result[0] += -0.10288450183797025;
                }
              } else {
                result[0] += 0.003912455207800394;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
                result[0] += 0.05205942962014685;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.400641441345215288) ) ) {
                        result[0] += -0.09156667805071478;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.238486170768738237) ) ) {
                          result[0] += 0.016518265904006298;
                        } else {
                          result[0] += -0.059912334741441975;
                        }
                      }
                    } else {
                      result[0] += 0.01911129067965968;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.10687490332557159;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                          result[0] += -0.06095469671793911;
                        } else {
                          result[0] += 0.04162402668690969;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.499747991561890537) ) ) {
                        result[0] += -0.05782949724447789;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.182021141052246982) ) ) {
                          result[0] += 0.004450344278019766;
                        } else {
                          result[0] += 0.07452884132771379;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.032948217075608424;
                  } else {
                    result[0] += -0.14196204012981292;
                  }
                }
              }
            }
          } else {
            result[0] += 0.01124371957192405;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
            result[0] += -0.017548607935937804;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
              result[0] += 0.0300246633730796;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.01917773626378525;
                } else {
                  result[0] += -0.07059506653337254;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.098033905029297763) ) ) {
                  result[0] += 0.0011383732706538175;
                } else {
                  result[0] += 0.09031973623927242;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += -0.0002712817026704246;
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.009871592880057536;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0526060088144617;
                } else {
                  result[0] += -0.10874555198452174;
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                    result[0] += -0.08602475364746619;
                  } else {
                    result[0] += -0.017210551662390106;
                  }
                } else {
                  result[0] += 0.006665184001138448;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.12053841750182877;
            } else {
              result[0] += -0.001950089148173302;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.07895948154491532;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
              result[0] += -0.0033896573880480875;
            } else {
              result[0] += -0.01829731130300657;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.05206266174159775;
            } else {
              result[0] += 0.0010048760938049814;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.034917736137267744;
              } else {
                result[0] += 0.05901429672524108;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.030607855326184215;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                  result[0] += 0.02881310135568324;
                } else {
                  result[0] += -0.03336991916038864;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.934722661972046787) ) ) {
            result[0] += -0.015718981509787715;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.09192466968489789;
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.05771471745994116;
                } else {
                  result[0] += -0.01664787046494373;
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.08906092289283032;
              } else {
                result[0] += -0.012104501889062187;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.547126770019532138) ) ) {
              result[0] += -0.053752830554095465;
            } else {
              result[0] += 0.043189593191886894;
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
              result[0] += 0.016549862479808542;
            } else {
              result[0] += 0.1010353960116202;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)214.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
          result[0] += -0.005528798000260338;
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.01308787387967933;
            } else {
              result[0] += -0.04731225997635915;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.997585535049439365) ) ) {
              result[0] += -0.05208707426067349;
            } else {
              result[0] += 0.012143171413130656;
            }
          }
        }
      } else {
        result[0] += 0.004627822869340573;
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
            result[0] += -0.003418577309564188;
          } else {
            if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.07431524571219468;
            } else {
              result[0] += 0.024042116554757825;
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
                result[0] += 0.011901931866081336;
              } else {
                result[0] += -0.05478297933282611;
              }
            } else {
              result[0] += -0.04572022533214104;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.008420981278257678;
              } else {
                result[0] += 0.01631673267177999;
              }
            } else {
              result[0] += -0.03583513434044049;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58603620529174982) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += 0.015145709313080203;
            } else {
              result[0] += -0.01834293087941038;
            }
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)84.50000000000001421) ) ) {
                result[0] += -0.038889677256968284;
              } else {
                result[0] += 0.03369559556103433;
              }
            } else {
              result[0] += -0.054606174745199104;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10844659805298029) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.01588841479590733;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                          result[0] += -0.02750495832034297;
                        } else {
                          result[0] += 0.009266079511826914;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.378218650817871982) ) ) {
                          result[0] += -0.023469320652666877;
                        } else {
                          result[0] += -0.09335720908659761;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)78.50000000000001421) ) ) {
                        result[0] += 0.04966179834907547;
                      } else {
                        result[0] += 0.015824139866659294;
                      }
                    } else {
                      result[0] += 0.0024552992298387086;
                    }
                  }
                } else {
                  result[0] += -0.029593801547921763;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.03296290336026999;
                    } else {
                      result[0] += -0.05184205864460972;
                    }
                  } else {
                    result[0] += -0.03254492955841841;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0006071903438823829;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)49.50000000000000711) ) ) {
                            result[0] += 0.11147736013057566;
                          } else {
                            result[0] += 0.030523219464587293;
                          }
                        } else {
                          result[0] += 0.019850340344398726;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.025580888617373166;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.024478041580451883;
                        } else {
                          result[0] += -0.017033187835861326;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.020937302498816878;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0021410103750921883;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.023356870619170918;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                    result[0] += -0.05902507795784365;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.011348997499737593;
                    } else {
                      result[0] += -0.061171156288180796;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 1.939152990316679e-05;
            } else {
              result[0] += 0.02479257612482756;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1479225158691424) ) ) {
                  result[0] += -0.011595531314192312;
                } else {
                  result[0] += -0.060926105678213965;
                }
              } else {
                result[0] += -0.0013045361147603915;
              }
            } else {
              result[0] += 0.0010272816937536586;
            }
          } else {
            result[0] += 0.015760044589532878;
          }
        } else {
          result[0] += -0.015806755255062567;
        }
      } else {
        result[0] += 0.00539503130461224;
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            result[0] += -0.06584592218724876;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.378218650817871982) ) ) {
                  result[0] += -0.01474779331193074;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)226.5000000000000284) ) ) {
                    result[0] += -0.00957987958934449;
                  } else {
                    result[0] += 0.019414931083535123;
                  }
                }
              } else {
                result[0] += -0.048414722511596855;
              }
            } else {
              result[0] += -0.04162934226134011;
            }
          }
        } else {
          result[0] += 0.005709400452313422;
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.062164093649963004;
        } else {
          result[0] += -0.01920190843250884;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.002985037961840514;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += -0.0037706555917712084;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0160294370663089;
              } else {
                result[0] += -0.05313429883723045;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
            result[0] += -0.072654055667919;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.011772821128875052;
              } else {
                result[0] += -0.0416193637001985;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.610357046127320224) ) ) {
                result[0] += -0.05055073117309594;
              } else {
                result[0] += 0.026213208886408434;
              }
            }
          }
        }
      } else {
        result[0] += 0.007779489721813278;
      }
    } else {
      if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
          result[0] += -0.0014431292141309341;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.065660476684572089) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += 0.03397677255359191;
            } else {
              result[0] += -0.028677719355782158;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.06044993555307135;
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.023082492293698617;
              } else {
                result[0] += 0.06054569722815681;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.01664118833925941;
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.008268663162749898;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  result[0] += -0.023331350820538346;
                } else {
                  result[0] += -0.08712027060736656;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)194.5000000000000284) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.008439333086856711;
                    } else {
                      result[0] += -0.010190373415281798;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.731793165206910068) ) ) {
                      result[0] += 0.010300380015300153;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.003392525461722857;
                      } else {
                        result[0] += 0.08730379951469004;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0997242642347517;
                }
              } else {
                result[0] += 0.027406054208205268;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08260454436166537;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += 0.0036287974069147723;
                } else {
                  result[0] += -0.05806015694651504;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)84.50000000000001421) ) ) {
                  result[0] += 0.06949747732844622;
                } else {
                  result[0] += 0.02171663949237957;
                }
              } else {
                result[0] += 0.01890772016955784;
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0065763014955523575;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.551017761230469638) ) ) {
                    result[0] += -0.011986150006980188;
                  } else {
                    result[0] += 0.059887546191939506;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06276416778564631) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.954540252685547763) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.46012759208679288) ) ) {
                      result[0] += -0.06328024992828882;
                    } else {
                      result[0] += 0.04494830874542361;
                    }
                  } else {
                    result[0] += 0.05295468866469184;
                  }
                } else {
                  result[0] += 0.09129282326843945;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35055541992187678) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.847910165786744052) ) ) {
                result[0] += -0.03825559877181376;
              } else {
                result[0] += 0.005432585967861308;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.01038400839169011;
              } else {
                result[0] += -0.08532989191767304;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            result[0] += -0.06336836269874907;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                  result[0] += -0.01376014193821231;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)226.5000000000000284) ) ) {
                    result[0] += -0.007955572507927408;
                  } else {
                    result[0] += 0.020285727455479553;
                  }
                }
              } else {
                result[0] += -0.04476340151903845;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.389061450958252841) ) ) {
                result[0] += -0.0071468567930522186;
              } else {
                result[0] += -0.05407723821839342;
              }
            }
          }
        } else {
          result[0] += 0.00407938432086165;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
          result[0] += -0.06392290226731634;
        } else {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.07003959292474704;
          } else {
            result[0] += -0.005813067094246272;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
                result[0] += -0.0034812152614381742;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                  result[0] += -0.0017590092266686005;
                } else {
                  result[0] += -0.06260532041497195;
                }
              }
            } else {
              result[0] += 0.001028167714785919;
            }
          } else {
            result[0] += 0.013817615996681888;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.06476533447026987;
          } else {
            result[0] += -0.00995743804567908;
          }
        }
      } else {
        result[0] += 0.005135016680015242;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.624904632568361151) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.597130775451661044) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.871366024017334873) ) ) {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.003923926751121364;
                  } else {
                    result[0] += 0.02000629499475576;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.524927973747253862) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                            result[0] += -0.008507278580966345;
                          } else {
                            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)62.50000000000000711) ) ) {
                              result[0] += 0.007212553173303953;
                            } else {
                              result[0] += 0.04125908374891097;
                            }
                          }
                        } else {
                          result[0] += 0.08228130109860397;
                        }
                      } else {
                        result[0] += -0.006971707278700068;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                        result[0] += -0.0055271695344935885;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.749434947967529741) ) ) {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.958868026733400214) ) ) {
                              result[0] += -0.007836047226781113;
                            } else {
                              result[0] += 0.06303153008775317;
                            }
                          } else {
                            result[0] += 0.0947582030727233;
                          }
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.499213218688965732) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.001201295058598346;
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.109245061874390537) ) ) {
                                  result[0] += 0.01923844739673767;
                                } else {
                                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.666320323944092685) ) ) {
                                      result[0] += 0.030982974874140756;
                                    } else {
                                      result[0] += 0.10895560307284742;
                                    }
                                  } else {
                                    result[0] += 0.015465978156673133;
                                  }
                                }
                              } else {
                                result[0] += 0.005788421733821085;
                              }
                            }
                          } else {
                            result[0] += -0.02199147082011381;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.314458370208742011) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                          result[0] += -0.004808862059464715;
                        } else {
                          result[0] += 0.0804076427475858;
                        }
                      } else {
                        result[0] += 0.04270712568951612;
                      }
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.902117252349854404) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                          result[0] += -4.988118276594554e-05;
                        } else {
                          result[0] += -0.04579066535452456;
                        }
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)6.500000000000000888) ) ) {
                            result[0] += 0.022037171329008436;
                          } else {
                            result[0] += -0.04918149646808205;
                          }
                        } else {
                          result[0] += 0.0025641701011423963;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0008557189189989595;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
                result[0] += 0.0012877144452120178;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  result[0] += -0.0020307302172576716;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)67.50000000000001421) ) ) {
                      result[0] += -0.05064325011852342;
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += 0.02096692503941707;
                      } else {
                        result[0] += -0.01774270530647532;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)80.50000000000001421) ) ) {
                        result[0] += -0.09707224328066014;
                      } else {
                        result[0] += 0.021397807096022566;
                      }
                    } else {
                      result[0] += -0.06861840996886497;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.08769367094266974;
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.0443644209959273;
                } else {
                  result[0] += -0.010015323664245082;
                }
              } else {
                result[0] += -0.014152179825348646;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.874179124832154208) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)249.5000000000000284) ) ) {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.03540484737499332;
                    } else {
                      result[0] += 0.08005020715386905;
                    }
                  } else {
                    result[0] += -0.0083089767147381;
                  }
                } else {
                  result[0] += 0.013215196954475867;
                }
              } else {
                result[0] += 0.002094512651483951;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.959336042404175693) ) ) {
                result[0] += -0.004468406935318752;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.037386643057138756;
                } else {
                  result[0] += -0.08956452738916545;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                  result[0] += 0.02101531632973903;
                } else {
                  result[0] += -0.04242920946466097;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                  result[0] += 0.018258339292057377;
                } else {
                  result[0] += 0.08959132373514363;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)179.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.06335434868628805;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.417592287063599077) ) ) {
              result[0] += -0.010924649842754125;
            } else {
              result[0] += -0.03883158045318819;
            }
          }
        } else {
          result[0] += 0.02145575599256147;
        }
      }
    } else {
      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.010266158835777549;
      } else {
        result[0] += 0.001122468266355304;
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      result[0] += -0.002448780744931308;
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += -0.02055031188827395;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.09831865817580597;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.051500320434572089) ) ) {
              result[0] += -0.0028402391390945883;
            } else {
              result[0] += 0.03169197513623626;
            }
          }
        } else {
          result[0] += -0.06734611632875254;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.586156606674195224) ) ) {
      result[0] += -0.0028122918364392718;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.004881381988526279) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.588751316070558417) ) ) {
            result[0] += -0.002897885006395446;
          } else {
            result[0] += -0.1478812826596034;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.318498134613038886) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.036715820453888146;
            } else {
              result[0] += -0.09294880673968858;
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.11853435418817378;
            } else {
              result[0] += 0.0808729381300956;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.004288052395673985;
        } else {
          result[0] += -0.06693244948567237;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.01901475047707288;
    } else {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)29.50000000000000355) ) ) {
              result[0] += -0.03981177853244392;
            } else {
              result[0] += 0.017451097225928212;
            }
          } else {
            result[0] += 0.002747422787807739;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
            result[0] += 0.0004027996156460566;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.008416517170285102;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.012991086657026033;
                } else {
                  result[0] += -0.000719273495318441;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77165889739990412) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.013715192498243293;
                    } else {
                      result[0] += -0.042670529178311864;
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += -0.003254126886050694;
                    } else {
                      result[0] += 0.11010051148977174;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.610357046127320224) ) ) {
                    result[0] += 0.01504276756769345;
                  } else {
                    result[0] += -0.05523290271860051;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.461708784103394443) ) ) {
                  result[0] += -0.006762364592923765;
                } else {
                  result[0] += -0.04403527797409229;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.004238852132501901;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += 0.0072066805448095;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.132848501205445224) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                        result[0] += 0.004046922320690915;
                      } else {
                        result[0] += -0.03744000436956237;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
                        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.05301432505302968;
                          } else {
                            result[0] += -0.01963543003855578;
                          }
                        } else {
                          result[0] += -0.011008271770451952;
                        }
                      } else {
                        result[0] += 0.004661942446262873;
                      }
                    }
                  } else {
                    result[0] += 0.004453336573309724;
                  }
                } else {
                  result[0] += 0.008493209547378483;
                }
              }
            } else {
              result[0] += 0.009173369714693963;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
            result[0] += -0.0014721651711225156;
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.06968239581387088;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += 0.006763542255320706;
                  } else {
                    result[0] += -0.017105019366464337;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += 0.011244289890693615;
                } else {
                  result[0] += -0.00496307169683952;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)109.5000000000000142) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10098505020141779) ) ) {
                      if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.189289569854737216) ) ) {
                            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                              result[0] += -0.031501288203156334;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
                                result[0] += -0.022000794147883272;
                              } else {
                                result[0] += 0.025314554572063802;
                              }
                            }
                          } else {
                            result[0] += 0.04133696483448465;
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.601370334625245029) ) ) {
                            result[0] += -0.04195141143024675;
                          } else {
                            result[0] += 0.04424176207938724;
                          }
                        }
                      } else {
                        result[0] += 0.049842962378423415;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.455312013626099521) ) ) {
                        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.1056954852324068;
                        } else {
                          result[0] += 0.037369534355581414;
                        }
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.002295865760517687;
                        } else {
                          result[0] += 0.06306708068647311;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.06529435833004847;
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)113.5000000000000142) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.473832368850708896) ) ) {
                        result[0] += -0.004047248532734546;
                      } else {
                        result[0] += -0.04647286608790001;
                      }
                    } else {
                      result[0] += 0.059968227095043715;
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.447260618209839755) ) ) {
                          result[0] += 0.011311541735186129;
                        } else {
                          result[0] += -0.012657079754526427;
                        }
                      } else {
                        result[0] += -0.053997454211045716;
                      }
                    } else {
                      result[0] += 0.026015086709477803;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
                  result[0] += -0.0017982904097778548;
                } else {
                  result[0] += 0.014930037833644697;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)37.50000000000000711) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)31.50000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
            result[0] += -0.0061976271020022684;
          } else {
            result[0] += -0.05558953191132557;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.044662661509059634;
              } else {
                result[0] += 0.030698948191780276;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                result[0] += -0.023771167585338884;
              } else {
                result[0] += 0.10305585368664799;
              }
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.543220520019532138) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.497206687927246982) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.763591527938843662) ) ) {
                        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          result[0] += -0.06586758971401578;
                        } else {
                          result[0] += -0.007119760553886141;
                        }
                      } else {
                        result[0] += 0.015091896082180199;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.63762140274048029) ) ) {
                        result[0] += 0.017373321683322337;
                      } else {
                        result[0] += -0.012247441265282403;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                      result[0] += 0.017245064542473886;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
                        result[0] += -0.011343491654581796;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                          result[0] += -0.015483227924086632;
                        } else {
                          result[0] += -0.07437144336041836;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
                        result[0] += 0.0012860423647208965;
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.63308715820312678) ) ) {
                          result[0] += -0.018329445159480973;
                        } else {
                          result[0] += -0.05246593624026128;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                          if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.003126514943527059;
                            } else {
                              result[0] += 0.024452565849247522;
                            }
                          } else {
                            result[0] += 0.06916299565181953;
                          }
                        } else {
                          result[0] += -0.05221854040722987;
                        }
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.548691272735597479) ) ) {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
                                result[0] += -0.03713177014816846;
                              } else {
                                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += 0.006937033343431719;
                                } else {
                                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                                    result[0] += -0.025900582136037426;
                                  } else {
                                    result[0] += -0.1230136936304554;
                                  }
                                }
                              }
                            } else {
                              result[0] += -0.0021685615584858832;
                            }
                          } else {
                            result[0] += -0.0020250313000833155;
                          }
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.016632797101830386;
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += -0.008297635156954695;
                            } else {
                              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                                  result[0] += -0.013582717959489966;
                                } else {
                                  result[0] += 0.012306214879946917;
                                }
                              } else {
                                result[0] += 0.03765887011776772;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.725620865821838823) ) ) {
                      result[0] += -0.05951165919002586;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.03556795577640245;
                      } else {
                        result[0] += 0.008429457288124134;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.013512632546278564;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.05552116189529367;
                } else {
                  result[0] += 0.09653718130916128;
                }
              } else {
                result[0] += 0.013982364487064242;
              }
            }
          }
        }
      } else {
        result[0] += -0.011536484252560918;
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)38.50000000000000711) ) ) {
        result[0] += 0.02012352218347372;
      } else {
        result[0] += 0.0008243676594911387;
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        result[0] += -0.00034166379158209807;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
          result[0] += -0.002804960845342685;
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.60064363479614435) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
              result[0] += -0.04003458421211905;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.10005515725104891;
                  } else {
                    result[0] += 0.010844096491372999;
                  }
                } else {
                  result[0] += -0.05471549974444942;
                }
              } else {
                result[0] += -0.021226271892351788;
              }
            }
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.060305353950153864;
            } else {
              result[0] += 0.016755359906989547;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.016205472854738768;
          } else {
            result[0] += -0.061339404388749456;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)9.167253971099855292) ) ) {
            result[0] += -0.07187808438312443;
          } else {
            result[0] += 0.1712311244170214;
          }
        }
      } else {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.497206687927246982) ) ) {
              result[0] += 0.08021203170244393;
            } else {
              result[0] += -0.03746465184975636;
            }
          } else {
            result[0] += -0.10850921252175559;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.400641441345215288) ) ) {
            result[0] += -0.06417445897207615;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)103.5000000000000142) ) ) {
              result[0] += 0.016141780148335772;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.07943695679622545;
              } else {
                result[0] += -0.0038171455476325167;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      result[0] += -7.811643661854507e-05;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338887453079224521) ) ) {
          result[0] += 0.007017209914496001;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
              result[0] += -0.059759096476193624;
            } else {
              result[0] += 0.022084456392264792;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0429112000866585;
            } else {
              result[0] += 0.01558298118849525;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.0010575603259542995;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)185.5000000000000284) ) ) {
                result[0] += -0.014609123909101303;
              } else {
                result[0] += -0.08868667956925683;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.022447375294318235;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.007488643543160755;
                    } else {
                      result[0] += -0.05037928940249131;
                    }
                  } else {
                    result[0] += 0.04348614570514351;
                  }
                } else {
                  result[0] += -0.010785198209915334;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.00502052541622611;
                } else {
                  result[0] += 0.03031153239546815;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)115.5000000000000142) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.831219434738160068) ) ) {
                  result[0] += -0.01468253920130503;
                } else {
                  result[0] += 0.0728115143325273;
                }
              } else {
                result[0] += 0.06929383189006848;
              }
            } else {
              result[0] += -0.047319877261910066;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.010563661988682853;
            } else {
              result[0] += 0.019136672943668734;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
            result[0] += 0.007573584206747207;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += -0.04244785137769392;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.821564435958863193) ) ) {
                result[0] += -0.0001334336019653163;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0773636847929218;
                } else {
                  result[0] += 0.009173706127697403;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.386624813079835761) ) ) {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.11098730389664667;
              } else {
                result[0] += -0.028326222034989093;
              }
            } else {
              result[0] += -0.006133042450985833;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.02881453091441076;
                  } else {
                    result[0] += -0.01260709564815229;
                  }
                } else {
                  result[0] += -0.05816386018461746;
                }
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.01691947493726316;
                } else {
                  result[0] += -0.07230001859095264;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.009311231925779201;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.34467267990112482) ) ) {
                  result[0] += -0.03326978556165596;
                } else {
                  result[0] += 0.012035604247862318;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += 0.0003490161411494404;
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.059420347213746005) ) ) {
              result[0] += -0.0498565292277721;
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.014280257767716402;
                } else {
                  result[0] += -0.08727965582197474;
                }
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.89399480819702326) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                      result[0] += -0.022764171366385293;
                    } else {
                      result[0] += 0.13223796073003927;
                    }
                  } else {
                    result[0] += -0.02991913795538653;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.21802663803100764) ) ) {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.238486170768738237) ) ) {
                          result[0] += -0.005749610467951233;
                        } else {
                          result[0] += -0.08213796201637576;
                        }
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.014572931754517427;
                        } else {
                          result[0] += -0.03013068360189476;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
                        result[0] += 0.0012767124701614013;
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                          result[0] += 0.0013516627803617206;
                        } else {
                          result[0] += -0.02514360590206312;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.02218367489762027;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.45601701736450373) ) ) {
              result[0] += -0.07047004256932814;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.10226969042916718;
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)9.500000000000001776) ) ) {
                    result[0] += 0.013779057929985303;
                  } else {
                    result[0] += -0.0636327046078121;
                  }
                } else {
                  result[0] += 0.02016322983498243;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += -0.018973520223256757;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += -0.0421318658984558;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += -0.09702945736160329;
            } else {
              result[0] += 0.008390636289920083;
            }
          }
        } else {
          result[0] += -0.06416884467838208;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0025847687307559795;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.06353464745146528;
                } else {
                  result[0] += 0.005968422188620876;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.99033999443054288) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                    result[0] += 0.03465826893501031;
                  } else {
                    result[0] += -0.04106312591247016;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    result[0] += 0.020314571010216276;
                  } else {
                    result[0] += 0.08105474252345722;
                  }
                }
              }
            }
          } else {
            result[0] += 0.02847233532829049;
          }
        } else {
          result[0] += -0.014027423779282684;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.17027091979980646) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)18.50000000000000355) ) ) {
            result[0] += -0.02473436671200462;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.00383058405373686;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                    result[0] += -0.01010270076876635;
                  } else {
                    result[0] += 0.009337580282610414;
                  }
                } else {
                  result[0] += -0.047592796253633475;
                }
              }
            } else {
              result[0] += -0.011866424676976274;
            }
          }
        } else {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.021347467549758196;
          } else {
            result[0] += 0.01260315830510035;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.015816640449149103;
          } else {
            result[0] += -0.04405161929080326;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.441221237182618076) ) ) {
            result[0] += -0.013219665159132083;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.02272748627009585;
              } else {
                result[0] += -0.1264941945344925;
              }
            } else {
              result[0] += 0.030406975867606007;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)236.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.473471879959107333) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.004244100511616027;
                } else {
                  result[0] += -0.024017438903347982;
                }
              } else {
                if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.033684518384631;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.890260934829712802) ) ) {
                    result[0] += -0.04276512763876328;
                  } else {
                    result[0] += 0.03365071468015884;
                  }
                }
              }
            } else {
              result[0] += 0.01985644768369132;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
              result[0] += -0.004616340039247493;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += -0.06945117749500872;
                } else {
                  result[0] += 0.0021632419377226667;
                }
              } else {
                result[0] += 0.021046951085436083;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
            result[0] += -0.054831862812758675;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.025057968285249418;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.05146196353041741;
                } else {
                  result[0] += 0.013732156948955127;
                }
              } else {
                result[0] += -0.07278288384876151;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)37.50000000000000711) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += 0.04765954812360597;
                } else {
                  result[0] += -0.0243217495330129;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += -0.01841434068958676;
                } else {
                  result[0] += -0.08352351803012452;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                result[0] += 0.009422015908543234;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.02242517871312008;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.004895534287957859;
                    } else {
                      result[0] += -0.020409243210201402;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
                      result[0] += -0.003347843237517455;
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += -0.005129476378301187;
                      } else {
                        result[0] += 0.04956159297568674;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.002010437327534192;
          }
        } else {
          if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.039434667475204785;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.040126444919520186;
            } else {
              result[0] += 0.06531250756106761;
            }
          }
        }
      } else {
        result[0] += 0.0029862006192365627;
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            result[0] += -0.06192576905102862;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.940192461013794833) ) ) {
                result[0] += -0.008762609768443888;
              } else {
                result[0] += 0.0090743928351583;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                result[0] += -0.008145487127937088;
              } else {
                result[0] += -0.05178576154954591;
              }
            }
          }
        } else {
          result[0] += 0.004796048098278212;
        }
      } else {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.062295177358986556;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
            result[0] += -0.07193942778865275;
          } else {
            result[0] += -0.012902858772973577;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      result[0] += -0.0001133995406644903;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.716979026794434482) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
            result[0] += -0.026811127164340504;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
              result[0] += 0.009187659158033832;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.352615833282471591) ) ) {
                result[0] += -0.042835705247887455;
              } else {
                result[0] += 0.00266016080670284;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03538198986403485;
              } else {
                result[0] += 0.010425307942719357;
              }
            } else {
              result[0] += 0.047155753886364664;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
              result[0] += 0.018946959669975786;
            } else {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.050941082703561624;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.06732626539391261;
                  } else {
                    result[0] += 0.00987879863053085;
                  }
                } else {
                  result[0] += 0.03148518064905216;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)196.5000000000000284) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.03733407346111006;
                  } else {
                    result[0] += -0.001644770491944838;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.048523522187102874;
                  } else {
                    result[0] += -0.011556818550866706;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.02879098905518035;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                      result[0] += 0.0060209906775208705;
                    } else {
                      result[0] += -0.024418153020819816;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.0359025992507034;
                  } else {
                    result[0] += 0.007622945569499326;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                result[0] += 0.09296804613169302;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.07742834742183413;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.023187788255028527;
                    } else {
                      result[0] += -0.06909682728964267;
                    }
                  }
                } else {
                  result[0] += -0.07803838426839724;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                result[0] += 0.033016917700081576;
              } else {
                result[0] += -0.009916729920470704;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                    result[0] += 0.03146982035926918;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                        result[0] += 0.006168965156117452;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.153832674026490146) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                            result[0] += 0.0028777058700316174;
                          } else {
                            result[0] += -0.07329738096630568;
                          }
                        } else {
                          result[0] += -0.05305667942859345;
                        }
                      }
                    } else {
                      result[0] += -0.086555021384324;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                      result[0] += 0.0005113659479506797;
                    } else {
                      result[0] += 0.041145379071448225;
                    }
                  } else {
                    result[0] += 0.01908798951834901;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)16.50000000000000355) ) ) {
                  result[0] += 0.01930321723557752;
                } else {
                  result[0] += -0.012810502249947738;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
              result[0] += -0.0352410109849051;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                result[0] += 0.012791290149507448;
              } else {
                result[0] += -0.04114638008185976;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
              result[0] += 0.029882506545549817;
            } else {
              result[0] += -0.06373233296940164;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.473832368850708896) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += -0.0002622469307537133;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += 0.00040525475011728975;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.08442185898172837;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.102609157562256748) ) ) {
                result[0] += -0.04621898512684188;
              } else {
                result[0] += 0.02337288447007038;
              }
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22767019271850764) ) ) {
                  result[0] += -0.06057713473483162;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += -0.07568204839204816;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.04429251740492162;
                    } else {
                      result[0] += 0.0056287347583772835;
                    }
                  }
                }
              } else {
                result[0] += -0.009563723393301159;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.98273515701294123) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.009394522489932222;
                  } else {
                    result[0] += 0.09878269352168667;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                    result[0] += 0.07467978967929251;
                  } else {
                    result[0] += -0.07533862612264786;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
                  result[0] += -0.05911015528763021;
                } else {
                  result[0] += 0.024495580159410224;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
        result[0] += -0.04356318240944739;
      } else {
        result[0] += -0.0109844969337101;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.0005130306818446727;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.767332553863526279) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.015618632282218986;
              } else {
                result[0] += -0.0050779870588386405;
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.045450652773432174;
              } else {
                result[0] += -0.06289039415452347;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
              result[0] += -0.0033369184661935906;
            } else {
              result[0] += -0.037566265923783114;
            }
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.962127923965454546) ) ) {
            result[0] += -0.05290247686647523;
          } else {
            result[0] += -0.01944208041990333;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.378218650817871982) ) ) {
          result[0] += -0.00019709038167373358;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
              result[0] += 0.0042304553433705925;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.352615833282471591) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.065660476684572089) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.014472897413409735;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.002163424490106805;
                    } else {
                      result[0] += -0.05133344294342948;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.07623525877874744;
                  } else {
                    result[0] += 0.0006254204535911576;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.06076953939441007;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.06045074309820967;
                  } else {
                    result[0] += 0.015052144220594157;
                  }
                }
              }
            }
          } else {
            result[0] += -0.04277450404975703;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.726826429367066318) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0009479142476665188;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80454301834106623) ) ) {
              result[0] += -0.030609389923156052;
            } else {
              result[0] += 0.008044898163213913;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += 0.002723437005177177;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
                result[0] += -0.012996975503449887;
              } else {
                result[0] += -0.0646414843400183;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += 0.008945860800149967;
                  } else {
                    result[0] += 0.046568627825656767;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.009918584012389141;
                  } else {
                    result[0] += -0.058398779390997746;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.109245061874390537) ) ) {
                  result[0] += 0.001310622536808817;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.02077286996131131;
                  } else {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.03975705676954791;
                    } else {
                      result[0] += 0.13815058256442822;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.004531091137987279;
                } else {
                  result[0] += -0.05015498124546184;
                }
              } else {
                result[0] += 0.008311436502531628;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.716979026794434482) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.038886519025976535;
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)123.5000000000000142) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.00012505453833538878;
            } else {
              result[0] += -0.0309294161409503;
            }
          } else {
            result[0] += 0.009127221990581878;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.385823249816895419) ) ) {
          result[0] += 0.02028744686561069;
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.04927905893852586;
          } else {
            result[0] += 0.011965582348231925;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)196.5000000000000284) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                result[0] += -0.0028532883625269297;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.046105237208572805;
                } else {
                  result[0] += -0.00996324919208494;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -3.2624430239883716e-05;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02435510452958725;
                } else {
                  result[0] += -0.053615164434548473;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.07549674950735435;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.028621453982822793;
                } else {
                  result[0] += -0.06601361760731794;
                }
              }
            } else {
              result[0] += -0.07561010813599771;
            }
          }
        } else {
          if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.668153762817383701) ) ) {
              result[0] += 0.04089869408575539;
            } else {
              result[0] += 0.007412446360380275;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
              result[0] += 0.006345645825746699;
            } else {
              result[0] += -0.006898658575284506;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
            result[0] += -0.0322548024128756;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
              result[0] += 0.010469342980839503;
            } else {
              result[0] += -0.03884582188315469;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
            result[0] += 0.026358750497794;
          } else {
            result[0] += -0.05956390161156214;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += 0.0022611575272366265;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.019624204897708962;
            } else {
              result[0] += 0.002885115577257792;
            }
          } else {
            result[0] += -0.06276064210924856;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.080862283706665927) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)146.5000000000000284) ) ) {
              result[0] += -0.06378820258108803;
            } else {
              result[0] += 0.013283979175117309;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += -0.018818082839152393;
                } else {
                  result[0] += 0.029617225740628486;
                }
              } else {
                result[0] += -0.004525575260519344;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
                result[0] += -0.0126226887430724;
              } else {
                result[0] += -0.07048504969685057;
              }
            }
          }
        } else {
          result[0] += -0.028407523345708924;
        }
      }
    } else {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.06107812527391171;
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += 0.01370235179249735;
              } else {
                result[0] += -0.07866870237018819;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                  result[0] += 0.008902943002476996;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07690525551115296;
                  } else {
                    result[0] += 0.007154111660711416;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.0477993581964896;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                    result[0] += -0.018799278337996835;
                  } else {
                    result[0] += 0.08307635127761065;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.000668633974634345;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.28360033035278498) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0064945207169317;
                } else {
                  result[0] += 0.034708990477471016;
                }
              } else {
                result[0] += 0.018597331080623374;
              }
            } else {
              result[0] += 0.018246332978161844;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.011952007970255994;
            } else {
              result[0] += -0.06302216902643613;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.00916569561994771;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)155.5000000000000284) ) ) {
                result[0] += -0.006715752196658688;
              } else {
                result[0] += -0.03567678650022681;
              }
            }
          } else {
            result[0] += 0.005868866899179644;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.467370748519898349) ) ) {
      result[0] += 0.004883892326644875;
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03158230076271622;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.012144150730698793;
                } else {
                  result[0] += 0.022401729034584802;
                }
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
                    result[0] += 0.02015856863722214;
                  } else {
                    result[0] += 0.06539526199753123;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                    result[0] += -0.015852109459949425;
                  } else {
                    result[0] += 0.04111141118706547;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.824383735656740058) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.009052888812639995;
              } else {
                result[0] += -0.05793640032109069;
              }
            } else {
              result[0] += -0.06120322235921674;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
            result[0] += -0.0040528408599969385;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.03389187240104052;
              } else {
                result[0] += 0.03496237261857601;
              }
            } else {
              result[0] += -0.05597422492894577;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
              result[0] += 0.018774229172611984;
            } else {
              result[0] += 0.07018692900387928;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
              result[0] += -0.005789656403290824;
            } else {
              result[0] += -0.07117383420305746;
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.05165803986217548;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.447260618209839755) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  result[0] += -0.04363377747690147;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                    result[0] += 0.004779365436484407;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
                      result[0] += -0.02411677431841507;
                    } else {
                      result[0] += 0.04503045156776389;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.027745770937469556;
                    } else {
                      result[0] += 0.07299392390035551;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                      result[0] += 0.050624781456967784;
                    } else {
                      result[0] += -0.0652143868652617;
                    }
                  }
                } else {
                  result[0] += 0.11274060291836713;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1479225158691424) ) ) {
              result[0] += 0.024143745556939534;
            } else {
              result[0] += -0.011098131154800319;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)6.500000000000000888) ) ) {
      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
        result[0] += 0.00018689014630118812;
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.134879350662232333) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.455312013626099521) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.21142535955986827;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                result[0] += 0.04137368250650913;
              } else {
                result[0] += -0.003923619122205167;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.0008331189994850287;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                result[0] += -0.053445080220293754;
              } else {
                result[0] += 0.04589555271017284;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.535362005233765537) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.010066330969686604;
                } else {
                  result[0] += 0.06348182250225792;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.236541748046876776) ) ) {
                  result[0] += -0.05685624973202982;
                } else {
                  result[0] += 0.00837689000662965;
                }
              }
            } else {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.07537699075457646;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
                      result[0] += -0.02632103948482098;
                    } else {
                      result[0] += 0.062906906874364;
                    }
                  }
                } else {
                  result[0] += -0.06034245020218994;
                }
              } else {
                result[0] += -0.07106045421708765;
              }
            }
          } else {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                  result[0] += 0.07988267027860389;
                } else {
                  result[0] += 0.03845520798799557;
                }
              } else {
                result[0] += -0.08214254302145718;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
                result[0] += 0.054805635478133746;
              } else {
                result[0] += -0.02992272794452231;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.028894671845455314;
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
          result[0] += 0.0012037844628811829;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.05314775604755423;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.535362005233765537) ) ) {
                  result[0] += -0.0985953186201271;
                } else {
                  result[0] += 0.012512074130871312;
                }
              }
            } else {
              result[0] += -0.07857358018972228;
            }
          } else {
            result[0] += -0.002049741363917802;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += -0.0009558615075514268;
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39368534088134943) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.032517189371925895;
                      } else {
                        result[0] += 0.144359593290301;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.795426130294800249) ) ) {
                        result[0] += -0.008344511354956565;
                      } else {
                        result[0] += 0.05218889277317704;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0884132385253924) ) ) {
                        result[0] += -0.11979841609140479;
                      } else {
                        result[0] += 0.06349234053660391;
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.0456538320901912;
                      } else {
                        result[0] += 0.04742906524019771;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81479597091674982) ) ) {
                    result[0] += -0.056686148605603705;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.07808550284662806;
                      } else {
                        result[0] += -0.0524368634531471;
                      }
                    } else {
                      result[0] += -0.03583533797075273;
                    }
                  }
                }
              } else {
                result[0] += -0.03073443484539198;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.84662961959839045) ) ) {
                result[0] += -0.01601305246469625;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0434050799944416;
                } else {
                  result[0] += 0.018384325781254734;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.45601701736450373) ) ) {
              result[0] += -0.06189742549836528;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.010254204555685454;
              } else {
                result[0] += -0.0423884879009616;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.210062026977539951) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.055318353253738975;
            } else {
              result[0] += 0.0008625823093113208;
            }
          } else {
            result[0] += 0.015323389503499006;
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.095905801848651;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.296216011047365058) ) ) {
                  result[0] += -0.0073905749450465415;
                } else {
                  result[0] += -0.07733631019354853;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.038761009362303527;
              } else {
                result[0] += -0.10424450271146445;
              }
            }
          } else {
            result[0] += -0.007087136036494603;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += -0.03797240001995689;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.400641441345215288) ) ) {
              result[0] += -0.08807426916032352;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                  result[0] += 0.022222439012412006;
                } else {
                  result[0] += 0.09988792625380657;
                }
              } else {
                result[0] += 0.0007009716023296024;
              }
            }
          }
        } else {
          result[0] += -0.06131984058924315;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)50.50000000000000711) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.052345187835917475;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.012838707075852046;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.0011304343193428478;
                    } else {
                      result[0] += 0.07193467384746807;
                    }
                  } else {
                    result[0] += -0.005267324179560027;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05081367492675959) ) ) {
                      result[0] += 0.00036149732248738067;
                    } else {
                      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.003206433200124949;
                        } else {
                          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.012072274648890725;
                          } else {
                            result[0] += 0.05942491383214941;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                          result[0] += -0.053198616101740974;
                        } else {
                          result[0] += 0.10304693297400205;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.03342122221408398;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.08799420200387331;
                      } else {
                        result[0] += -0.0015640680523406497;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                    result[0] += 0.02457715356719071;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.014542497771580508;
                    } else {
                      result[0] += 0.08949667792514314;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05081367492675959) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.04970563597838354;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.524927973747253862) ) ) {
                            result[0] += 0.006547434749071078;
                          } else {
                            result[0] += 0.08339917144956471;
                          }
                        } else {
                          result[0] += -0.010285570880336872;
                        }
                      }
                    } else {
                      result[0] += -0.09749298565385525;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59605169296264826) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.04868794068431913;
                        } else {
                          result[0] += 0.009869608042406702;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.03160623206625228;
                        } else {
                          result[0] += -0.0004168343278259581;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.1259701542284409;
                      } else {
                        result[0] += 0.042351141831307455;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
                    result[0] += -0.05908292176407594;
                  } else {
                    result[0] += -0.11687602283078238;
                  }
                }
              }
            }
          } else {
            result[0] += 0.026320967293073885;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
            result[0] += 0.0007613676430209083;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.09405650041783348;
              } else {
                result[0] += -0.016233951774085716;
              }
            } else {
              result[0] += -0.09631712310496364;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)39.50000000000000711) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.0020299869389644697;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.10413723801432379;
                    } else {
                      result[0] += -0.04708291640643499;
                    }
                  } else {
                    result[0] += 0.013534821268536366;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                      result[0] += -0.0208733082981399;
                    } else {
                      result[0] += -0.07411484823330546;
                    }
                  } else {
                    result[0] += 0.016594165029419346;
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)27.50000000000000355) ) ) {
                  result[0] += -0.016176423633947737;
                } else {
                  result[0] += -0.06016938951040244;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10844659805298029) ) ) {
                result[0] += -0.0031766611140266405;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)13.50000000000000178) ) ) {
                  result[0] += 0.06458202058318692;
                } else {
                  result[0] += -0.002869294411639704;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.82132816314697443) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
              result[0] += -0.007796516539528238;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.182021141052246982) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.012045901972102499;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)45.50000000000000711) ) ) {
                    result[0] += -0.015754647452157346;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                      result[0] += 0.18297321357711788;
                    } else {
                      result[0] += 0.0545319343728318;
                    }
                  }
                }
              } else {
                result[0] += 0.012851442137342817;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.015360784416299284;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.12432729903917922;
              } else {
                result[0] += -0.02274119706163022;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        result[0] += -0.0007257453743953556;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
              result[0] += -0.004467724862872218;
            } else {
              result[0] += -0.018384751845154163;
            }
          } else {
            result[0] += -0.023968774213040053;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
            result[0] += 0.00478846174394679;
          } else {
            result[0] += -0.02819331602098213;
          }
        }
      }
    }
  } else {
    result[0] += 0.0007406382525536932;
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.400641441345215288) ) ) {
            result[0] += -0.05163653704345831;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
              result[0] += -0.09250956373814874;
            } else {
              result[0] += 0.06116841371708606;
            }
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += -0.03071274107108687;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)22.50000000000000355) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.132412433624269354) ) ) {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.177185058593750444) ) ) {
                          result[0] += 0.0725574304950976;
                        } else {
                          result[0] += -0.0010219752775681757;
                        }
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.007508105231265091;
                        } else {
                          result[0] += -0.02542595461473092;
                        }
                      }
                    } else {
                      result[0] += 0.0007424245488975397;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.009660905156027955;
                    } else {
                      result[0] += 0.013849059451705016;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.254884481430054599) ) ) {
                      result[0] += 0.0022691063572689643;
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                          result[0] += -0.024287610236811398;
                        } else {
                          result[0] += -0.14924813965854142;
                        }
                      } else {
                        result[0] += -0.03657491350593619;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.444577455520630771) ) ) {
                      result[0] += -0.005278061112047514;
                    } else {
                      result[0] += 0.032457321350270095;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.50511837005615412) ) ) {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0518401204784014;
                    } else {
                      result[0] += -0.0031998133344650855;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.08187784659379328;
                      } else {
                        result[0] += -0.01081384129540186;
                      }
                    } else {
                      result[0] += -0.107326571600308;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0024420077723854607;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.905434608459474433) ) ) {
                          result[0] += -0.039965847581600045;
                        } else {
                          result[0] += -0.010662693308964777;
                        }
                      } else {
                        result[0] += 0.0111893118558236;
                      }
                    } else {
                      result[0] += 0.012315146497520942;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.716979026794434482) ) ) {
                  result[0] += 0.0372917045674541;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.014286699736027557;
                    } else {
                      result[0] += -0.021932368827116723;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                      result[0] += 0.016520693254412255;
                    } else {
                      result[0] += -0.06992098872547917;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
                      result[0] += 0.0407133931673964;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.810120582580567294) ) ) {
                        result[0] += 0.05904562738288809;
                      } else {
                        result[0] += -0.0421046326823806;
                      }
                    }
                  } else {
                    result[0] += -0.0059276450312491895;
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += 0.0906060821620569;
                  } else {
                    result[0] += 0.0296425571763206;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
              result[0] += 0.03831726158795131;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.352615833282471591) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.03344903909567894;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.10958961112002202;
                  } else {
                    result[0] += -0.04564883082396132;
                  }
                }
              } else {
                result[0] += 0.07521591222397417;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.0635179002325819;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.08615827560424982) ) ) {
                result[0] += 0.04403123568071944;
              } else {
                result[0] += 0.1002024614451057;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
              result[0] += -0.10448537613005973;
            } else {
              result[0] += -0.017422542065325508;
            }
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.09342704379413927;
              } else {
                result[0] += -0.008470786111533405;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                result[0] += 0.004481984108365619;
              } else {
                result[0] += 0.03359666108195681;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.0006277669503442853;
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.109050035476685458) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -9.616317251381781e-05;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += -0.0009862263840959378;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.04180624235809003;
              } else {
                result[0] += 0.022448819225314394;
              }
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.028714826122955783;
              } else {
                result[0] += -0.008928616253286323;
              }
            }
          } else {
            result[0] += -0.026348438671953585;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
        result[0] += -0.06916924139745909;
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)227.5000000000000284) ) ) {
          result[0] += -0.021494492127820005;
        } else {
          result[0] += 0.022491007511968936;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03244114705745457;
              } else {
                result[0] += 0.10194102678836002;
              }
            } else {
              result[0] += -0.003119924588464577;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.060116140058222536;
            } else {
              result[0] += 0.008050625643565856;
            }
          }
        } else {
          result[0] += -0.0007854736310738766;
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)163.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.350257158279419833) ) ) {
              result[0] += -0.010135656342966379;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.028796011973898333;
              } else {
                result[0] += -0.07957337055334159;
              }
            }
          } else {
            if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.013226127910013236;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.010638097646690688;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.09711860458501403;
                      } else {
                        result[0] += -0.002478902564908206;
                      }
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.09511202406843662;
                      } else {
                        result[0] += -0.049952912213179185;
                      }
                    }
                  }
                } else {
                  result[0] += -0.07152392187693035;
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
                    result[0] += -5.760540760205363e-05;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                      result[0] += 0.007229372069970232;
                    } else {
                      result[0] += -0.04835307967552559;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.25500679016113459) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.07157089504160179;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.256982564926148349) ) ) {
                        result[0] += -0.014240850537680129;
                      } else {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)12.50000000000000178) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.97070193290710538) ) ) {
                            result[0] += -0.06372194033223598;
                          } else {
                            result[0] += 0.09882713463218419;
                          }
                        } else {
                          result[0] += 0.011894097755659291;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.07960096051396287;
                    } else {
                      result[0] += 0.03017092724394372;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)212.5000000000000284) ) ) {
            result[0] += -0.03452613739957365;
          } else {
            result[0] += -0.002643483346731418;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.396947860717774326) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
              result[0] += -0.015120414639877839;
            } else {
              result[0] += 0.002956245353890569;
            }
          } else {
            result[0] += 0.0012354785875050365;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.03832255098724728;
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.004521155074767078;
            } else {
              result[0] += -0.03797547287711165;
            }
          }
        }
      } else {
        result[0] += -0.0034182108627546317;
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += -0.02157167563565516;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.021836024136338658;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.040432760930756366;
              } else {
                result[0] += -0.0756552092060043;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.03255220383723136;
              } else {
                result[0] += 0.002598558942964598;
              }
            } else {
              result[0] += 0.016928911894487843;
            }
          } else {
            result[0] += -0.04023750923104783;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
            result[0] += 0.0025918641269707756;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)204.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.022538185119630683) ) ) {
                result[0] += -0.020749891895447722;
              } else {
                result[0] += 0.04128859182061961;
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.0034875884025491253;
              } else {
                result[0] += 0.05202630825583265;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)174.5000000000000284) ) ) {
              result[0] += 0.0038596039051277627;
            } else {
              result[0] += 0.033764162348339344;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.997585535049439365) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += -0.04220760592095902;
                } else {
                  result[0] += -0.00420354561607958;
                }
              } else {
                result[0] += 0.0026528116167894487;
              }
            } else {
              result[0] += -0.027232806550295915;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
            result[0] += -0.0026330174849460604;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)113.5000000000000142) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                  result[0] += 0.02458035596998211;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.1153477375329787;
                  } else {
                    result[0] += -0.050346602430695525;
                  }
                }
              } else {
                result[0] += 0.007381453993924174;
              }
            } else {
              result[0] += 0.0015464428366338268;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05081367492675959) ) ) {
            result[0] += -0.018318386920605927;
          } else {
            result[0] += 0.016701002194649605;
          }
        }
      } else {
        result[0] += -0.016372571848559902;
      }
    }
  }
  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.0024966450034290813;
        } else {
          result[0] += 0.023570101963760342;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.09145545959472834) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.003910760354684154;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                result[0] += -0.03078422417903335;
              } else {
                result[0] += 0.005381409362706931;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.138333082199097124) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += -0.014381146540437838;
                } else {
                  result[0] += -0.05105870803146195;
                }
              } else {
                result[0] += -0.09920087121725496;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.01745305252657592;
            } else {
              result[0] += 0.009348155158616287;
            }
          } else {
            result[0] += 0.0709742844117998;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.313854932785035068) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)204.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
                result[0] += 0.011830074301215745;
              } else {
                result[0] += -0.027228105298049084;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
                      result[0] += -0.03398009207588854;
                    } else {
                      result[0] += 0.012979276698558091;
                    }
                  } else {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0638559401648174;
                    } else {
                      result[0] += -0.02829468360819893;
                    }
                  }
                } else {
                  result[0] += -0.07993728898098815;
                }
              } else {
                result[0] += -0.01686672808490425;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                result[0] += -0.019984536406796406;
              } else {
                result[0] += -0.0726819218120746;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0024228154014326905;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04380188166407624;
                  } else {
                    result[0] += 0.0020719097896422464;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.017667479280046288;
                  } else {
                    result[0] += 0.07013849877280406;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.006472352545687218;
            } else {
              result[0] += -0.07438680314975547;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.023659249983357944;
              } else {
                result[0] += -0.007400383122816597;
              }
            } else {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.061902782872042574;
              } else {
                result[0] += -0.0021140673661131926;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.026385049630341603;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.549646615982056552) ) ) {
            result[0] += -0.02210538909408534;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.01913905233513514;
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.008435718719368594;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.278867721557618964) ) ) {
                      result[0] += -0.029073919176909838;
                    } else {
                      result[0] += 0.14049064647980974;
                    }
                  } else {
                    result[0] += 0.06595794153817877;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.386624813079835761) ) ) {
                    result[0] += -0.09379514066466123;
                  } else {
                    result[0] += 0.04104830322691209;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)168.5000000000000284) ) ) {
            result[0] += 0.001443451185227057;
          } else {
            result[0] += 0.023886833279084776;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.037174106902833265;
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)179.5000000000000284) ) ) {
                      result[0] += -0.12242931868889986;
                    } else {
                      result[0] += 0.015907828309928755;
                    }
                  } else {
                    result[0] += 0.019017301864507105;
                  }
                } else {
                  result[0] += -0.020315938882771827;
                }
              }
            } else {
              result[0] += 0.0011822469535085287;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.031872749328615058) ) ) {
                result[0] += -0.004348238614446784;
              } else {
                result[0] += -0.0958489836370923;
              }
            } else {
              result[0] += -0.037855169130802;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.00013263544646951368;
            } else {
              result[0] += 0.09171162411921249;
            }
          } else {
            result[0] += -0.04570119011769056;
          }
        } else {
          result[0] += 0.02287999822141208;
        }
      }
    } else {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
          result[0] += -0.08187428985440262;
        } else {
          result[0] += -0.025450048713637482;
        }
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)160.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.03442314115714506;
            } else {
              result[0] += 0.010077952692915913;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
              result[0] += -0.06272553986799417;
            } else {
              result[0] += -0.008054740464214259;
            }
          }
        } else {
          result[0] += -0.04839432403536825;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.02414795313565085;
        } else {
          result[0] += 0.0005985572799546607;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.824383735656740058) ) ) {
            result[0] += -0.01153099030690693;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.004580548927574003;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.007031629419494572;
              } else {
                result[0] += -0.0732686029120216;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.375737190246582919) ) ) {
              result[0] += -0.009429544786608422;
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.049950394914457984;
              } else {
                result[0] += 0.006890967964074087;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.95229363441467374) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                      result[0] += -0.02723942864767133;
                    } else {
                      result[0] += -0.0016336807214485623;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                      result[0] += 0.07972015317875725;
                    } else {
                      result[0] += -0.04115394143154023;
                    }
                  }
                } else {
                  result[0] += -0.07365283999626752;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.04496475463071732;
                } else {
                  result[0] += 0.0033100753310897238;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01932286688611283;
              } else {
                result[0] += 0.0015604620381702008;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.03499417133076941;
        } else {
          result[0] += 0.002272276868168596;
        }
      } else {
        result[0] += -0.0028230627860229575;
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
            result[0] += 0.042028792205286825;
          } else {
            result[0] += -0.042008240430558334;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += 0.03677550858316637;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)70.50000000000001421) ) ) {
                  result[0] += -0.08598013693884687;
                } else {
                  result[0] += 0.09809075353528321;
                }
              } else {
                result[0] += -0.049591992905807486;
              }
            }
          } else {
            result[0] += -0.056653050027707146;
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
              result[0] += -0.06377099861962353;
            } else {
              result[0] += -0.0007810099647147461;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
              result[0] += 0.010482924351736856;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.25500679016113459) ) ) {
                result[0] += 0.045694064868391726;
              } else {
                result[0] += 0.0010452716231165347;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.549068689346314365) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                  result[0] += 0.002213485090242563;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += 0.007138121638359424;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.382196187973023349) ) ) {
                      result[0] += -0.009015154429683904;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.02709256905439797;
                        } else {
                          result[0] += 0.018742783398654216;
                        }
                      } else {
                        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                            result[0] += -0.014491513759701792;
                          } else {
                            result[0] += -0.09903149019756931;
                          }
                        } else {
                          result[0] += -0.02879669189486868;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.987184524536133701) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                      result[0] += 0.012709998405456528;
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.05587492590121626;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.970040798187256748) ) ) {
                          result[0] += -0.00956364036520678;
                        } else {
                          result[0] += 0.05237840825671586;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.03243122035662524;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.02147911933394836;
                  } else {
                    result[0] += 0.014263953868788258;
                  }
                }
              }
            } else {
              result[0] += -0.017976623871540198;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.422362327575684482) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += -0.007083163060654556;
              } else {
                result[0] += 0.0078029856576850645;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += 0.00465393761681763;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)115.5000000000000142) ) ) {
                  result[0] += 0.002478560514919588;
                } else {
                  result[0] += -0.05252802405055948;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
          result[0] += -0.002392219313444859;
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)113.5000000000000142) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.173316955566407138) ) ) {
                result[0] += 0.0076966270251866906;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.012312412476426868;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.12344697546817965;
                    } else {
                      result[0] += -0.06376301896018963;
                    }
                  } else {
                    result[0] += 0.008411669938094106;
                  }
                }
              }
            } else {
              result[0] += 0.0015723957079856053;
            }
          } else {
            result[0] += -0.011333437786237749;
          }
        }
      } else {
        result[0] += -0.014070631200128865;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.022209944199492154;
        } else {
          result[0] += 0.00048551431855461984;
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.212100267410279208) ) ) {
            result[0] += -0.013849234353964589;
          } else {
            result[0] += -0.04178079138305337;
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
              result[0] += -0.009065141021838663;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0207852283717884;
              } else {
                result[0] += -0.07443174595890541;
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.001403905624338293;
            } else {
              result[0] += -0.024230934476896993;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.032555414862238806;
        } else {
          result[0] += 0.0019688506177013695;
        }
      } else {
        result[0] += -0.0025811400689816485;
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
            result[0] += 0.04123952833303461;
          } else {
            result[0] += -0.04513109204889171;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
            result[0] += 0.011297852034624504;
          } else {
            result[0] += -0.051938528140723444;
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.590443611145021308) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.07337437974712806;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.10280810875291474;
                  } else {
                    result[0] += -0.006165542976328259;
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01737670955640674;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.399426221847535068) ) ) {
                      result[0] += -0.01941850151351207;
                    } else {
                      result[0] += 0.04082967638217019;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
                  result[0] += 0.015296006771137488;
                } else {
                  result[0] += -0.07305498146750826;
                }
              } else {
                result[0] += 0.02019272785292515;
              }
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.873524427413941318) ) ) {
              result[0] += -0.011312372059454085;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.02149554790892366;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                    result[0] += 0.024746604137844574;
                  } else {
                    result[0] += -0.08993729223759375;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.053044869696441455;
                  } else {
                    result[0] += 0.01764540435553101;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
              result[0] += 0.0024749760486981434;
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.003795926167070177;
                  } else {
                    result[0] += -0.04945539825734459;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.03870709397009983;
                  } else {
                    result[0] += 0.03859625387535181;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.236541748046876776) ) ) {
                    result[0] += -0.039971627838308404;
                  } else {
                    result[0] += 0.12323280339708127;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.980170249938965732) ) ) {
                      result[0] += 0.046519333982999594;
                    } else {
                      result[0] += 0.11294646964402447;
                    }
                  } else {
                    result[0] += -0.04113538744691497;
                  }
                }
              }
            }
          } else {
            result[0] += 0.005050376239865088;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.78399753570556818) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.137252807617188388) ) ) {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.003947292176829142;
              } else {
                result[0] += 0.017549128579720522;
              }
            } else {
              result[0] += -0.0388144207608849;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.09278297424316584) ) ) {
                result[0] += -0.09778757810527572;
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.1517297898295124;
                } else {
                  result[0] += -0.03111498664942271;
                }
              }
            } else {
              result[0] += -0.06490283961065811;
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)84.50000000000001421) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.005858507684492137;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)75.50000000000001421) ) ) {
                result[0] += -0.00543286412858807;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                  result[0] += -0.0065486594246485855;
                } else {
                  result[0] += -0.07135294379376926;
                }
              }
            }
          } else {
            result[0] += 0.0018943982107071046;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.847910165786744052) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.09278297424316584) ) ) {
              result[0] += -0.01758467878892928;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.06882500648498624) ) ) {
                result[0] += -0.05550328679313957;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                  result[0] += 0.11194449521250377;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                    result[0] += -0.18310480782105076;
                  } else {
                    result[0] += 0.1399877772713025;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
              result[0] += 0.01261943171569329;
            } else {
              result[0] += 0.11100733185505396;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            result[0] += -0.0005097189380840879;
          } else {
            result[0] += -0.03654322536680613;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += -0.007334065668593146;
      } else {
        result[0] += -0.0004717675444722155;
      }
    } else {
      result[0] += -0.02116200772644721;
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.883084774017335761) ) ) {
            result[0] += -0.005453980031089702;
          } else {
            result[0] += 0.049649417342257786;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.447260618209839755) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)224.5000000000000284) ) ) {
                    result[0] += -0.00667917667675127;
                  } else {
                    result[0] += 0.057018422721502204;
                  }
                } else {
                  result[0] += -0.04906873436412922;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.222574234008789951) ) ) {
                  result[0] += 0.016059956335473375;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.08540268791660985;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += 0.025413363096734976;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.10338261947271672;
                        } else {
                          result[0] += -0.020091393854063597;
                        }
                      } else {
                        result[0] += 0.04527587409183529;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.000307083129883701) ) ) {
                result[0] += -0.057931610237164914;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.831219434738160068) ) ) {
                  result[0] += -0.004886681196873779;
                } else {
                  result[0] += -0.06484652878835077;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.344720840454102451) ) ) {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.07215305989570703;
                } else {
                  result[0] += -0.021690325197637875;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                  result[0] += -0.025838309339896684;
                } else {
                  result[0] += -0.07080513719730404;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.012346000288116917;
                  } else {
                    result[0] += 0.03890790824896722;
                  }
                } else {
                  result[0] += -0.07692960436691738;
                }
              } else {
                result[0] += -0.036293470826659145;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)67.50000000000001421) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.280697107315064365) ) ) {
                result[0] += -0.031161061559605015;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.004552117931049458;
                } else {
                  result[0] += -0.0750123453954652;
                }
              }
            } else {
              result[0] += 0.012792450777803869;
            }
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.873524427413941318) ) ) {
              result[0] += -0.014290546739039251;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                result[0] += 0.015359092527979595;
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.08555827986842662;
                  } else {
                    result[0] += 0.05294321774974329;
                  }
                } else {
                  result[0] += -0.007971930485537355;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)72.50000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.200417995452881748) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                result[0] += 0.07874145681148281;
              } else {
                result[0] += 0.003848442266046562;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -0.1177467624994309;
              } else {
                result[0] += -0.03377161179326164;
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.975242614746095526) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.997585535049439365) ) ) {
                  result[0] += -0.028458823488437004;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.004881381988526279) ) ) {
                    result[0] += -0.02154037347516429;
                  } else {
                    result[0] += 0.0938670121754353;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
                      result[0] += -0.014080332046871003;
                    } else {
                      result[0] += 0.04613084111280607;
                    }
                  } else {
                    result[0] += 0.0921633661384346;
                  }
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.10279280136807159;
                  } else {
                    result[0] += 0.038922551865376354;
                  }
                }
              }
            } else {
              result[0] += 0.002314133991792063;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0033005318147714454;
          } else {
            result[0] += -0.019769898682512506;
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.0019344631078310787;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.45601701736450373) ) ) {
              result[0] += 0.024649717443289906;
            } else {
              result[0] += -0.005156022006876957;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.847910165786744052) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.019362834769764324;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += 0.02578401383470283;
                    } else {
                      result[0] += -0.07108392978053434;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.044887955284828356;
                    } else {
                      result[0] += 0.08867089478462781;
                    }
                  }
                } else {
                  result[0] += -0.16276872160598355;
                }
              } else {
                result[0] += 0.09485342386578365;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
              result[0] += -0.028062366090294184;
            } else {
              result[0] += 0.07251977628806115;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
            result[0] += -0.004133444290658301;
          } else {
            result[0] += -0.035142373724094164;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        result[0] += -0.0003809152374161321;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.016541026570365262;
            } else {
              result[0] += -0.05828366306973457;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                result[0] += -0.008689093912693796;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.005208261092491258;
                } else {
                  result[0] += -0.06562729314904309;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.011404818801561748;
                } else {
                  result[0] += -0.037290081799340256;
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.665046453475953037) ) ) {
                      result[0] += 0.009682263070588644;
                    } else {
                      result[0] += -0.05534104145390775;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                        result[0] += 0.001077862846536495;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.02592887792972053;
                          } else {
                            result[0] += -0.07001413767016522;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.975242614746095526) ) ) {
                            result[0] += 0.02222258729182803;
                          } else {
                            result[0] += -0.030565297173215968;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.013549069723402231;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.56219196319580256) ) ) {
                    result[0] += 0.0026854736983798894;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
                      result[0] += -0.024335374391744217;
                    } else {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.021513620483110965;
                      } else {
                        result[0] += 0.06381997595914592;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.975242614746095526) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.04500236918251604;
              } else {
                result[0] += 0.012043905647112909;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.011369034835197139;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.872101783752442294) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.221583127975464755) ) ) {
                    result[0] += -0.013370431100139683;
                  } else {
                    result[0] += -0.06636253983072338;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.256982564926148349) ) ) {
                    result[0] += 0.015587854491825873;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.062270963560406414;
                    } else {
                      result[0] += 0.004380514295323162;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)13.50000000000000178) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.06392112357159462;
              } else {
                result[0] += 0.031080975513546723;
              }
            } else {
              result[0] += -0.04509059670095647;
            }
          }
        }
      }
    } else {
      result[0] += 0.0001752042276256695;
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.549068689346314365) ) ) {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.602003335952759233) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.610145330429078037) ) ) {
                result[0] += 0.0059937937500121755;
              } else {
                result[0] += 0.04310771223308363;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.029564674028534223;
                } else {
                  result[0] += -0.0011712994567944818;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                  result[0] += -0.005419329272109769;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.09833610833609789;
                  } else {
                    result[0] += -0.023183810344050503;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.921924352645874468) ) ) {
                      result[0] += -0.07341318535969268;
                    } else {
                      result[0] += -0.21758564450600942;
                    }
                  } else {
                    result[0] += 0.003281153030780907;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)235.5000000000000284) ) ) {
                    result[0] += 0.0010992848128245946;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                      result[0] += 0.026992292222115546;
                    } else {
                      result[0] += -0.025182068176977646;
                    }
                  }
                }
              } else {
                result[0] += -0.012674527680143525;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.017087011476196314;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)252.5000000000000284) ) ) {
                  result[0] += 0.011121728793865608;
                } else {
                  result[0] += -0.0488765787502883;
                }
              }
            }
          }
        } else {
          result[0] += -0.01699818596470259;
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.422362327575684482) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0026821633522591958;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
                  result[0] += -0.0065879682719028445;
                } else {
                  result[0] += -0.03477047343427891;
                }
              }
            } else {
              result[0] += 0.008727014752575118;
            }
          } else {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.432135581970215732) ) ) {
                result[0] += 0.015164725702348876;
              } else {
                result[0] += -0.0794476316199004;
              }
            } else {
              result[0] += 0.010631114043487612;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
            result[0] += -0.0006297199035223439;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.045424807164811565;
            } else {
              result[0] += 0.074932544925757;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
          result[0] += -0.002078949565295598;
        } else {
          result[0] += 0.001745616090889398;
        }
      } else {
        result[0] += -0.011305068611156432;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
      result[0] += -0.007289211083030088;
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.396947860717774326) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.006448560275910015;
                  } else {
                    result[0] += -0.008379437963984838;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.810120582580567294) ) ) {
                    result[0] += 0.02790742909150956;
                  } else {
                    result[0] += -0.0037942641205919708;
                  }
                }
              } else {
                result[0] += -0.015929000109191384;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.01769117142908318;
                } else {
                  result[0] += -0.04547824593136078;
                }
              } else {
                result[0] += -0.005088431099384605;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.065660476684572089) ) ) {
                      if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.025828064808148223;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                          result[0] += -0.004498918280294562;
                        } else {
                          result[0] += -0.028927518540783927;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.003929950204909224;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                            result[0] += 0.007703420352560718;
                          } else {
                            result[0] += -0.014091308036635742;
                          }
                        } else {
                          result[0] += 0.027162454815803136;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.183091163635254794) ) ) {
                        result[0] += -0.0022801511247003813;
                      } else {
                        result[0] += -0.05746102225290918;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4982786178588885) ) ) {
                            result[0] += -0.0014073098044315637;
                          } else {
                            result[0] += -0.06152043743133271;
                          }
                        } else {
                          result[0] += -0.07394133319942105;
                        }
                      } else {
                        result[0] += 0.02838438932262793;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.388237953186036044) ) ) {
                        result[0] += -0.0014759297453530804;
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.058336352049480206;
                        } else {
                          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.028563158928392327;
                          } else {
                            result[0] += 0.015655663720643774;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.640130996704102451) ) ) {
                        result[0] += 0.006310274338311149;
                      } else {
                        result[0] += -0.02371104350371226;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.260223388671876776) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.32530093193054288) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
                              if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                                result[0] += -0.03952120644940898;
                              } else {
                                result[0] += 0.020688798612281227;
                              }
                            } else {
                              result[0] += 0.04240676629107124;
                            }
                          } else {
                            result[0] += -0.060984520870745754;
                          }
                        } else {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)17.50000000000000355) ) ) {
                            result[0] += -0.07202076444905707;
                          } else {
                            result[0] += -0.02898457092609237;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18779706954956232) ) ) {
                          result[0] += 0.0020301223914808587;
                        } else {
                          result[0] += 0.033430497577941976;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.014893864358086631;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.07496014014878823;
                          } else {
                            result[0] += 0.03155505527037977;
                          }
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                              result[0] += -0.09380828639477638;
                            } else {
                              result[0] += 0.017671093035803675;
                            }
                          } else {
                            result[0] += 0.037750923782826344;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
                  result[0] += -0.05304903975939867;
                } else {
                  if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0006565354996895398;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9916415214538592) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)50.50000000000000711) ) ) {
                            result[0] += -0.03881846514075618;
                          } else {
                            result[0] += -0.006991252134961275;
                          }
                        } else {
                          result[0] += -0.00820930535936039;
                        }
                      } else {
                        result[0] += -0.10090603037060784;
                      }
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)19.50000000000000355) ) ) {
                        result[0] += 0.037599769789613845;
                      } else {
                        result[0] += -0.0024962627465206343;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.051848633552176504;
                } else {
                  result[0] += -0.08882512849754032;
                }
              } else {
                result[0] += 0.01459612348835322;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.506659984588624823) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.0202056579818296;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.783307552337647373) ) ) {
                      result[0] += 0.011786213577181348;
                    } else {
                      result[0] += -0.039723730311042114;
                    }
                  } else {
                    result[0] += 0.026156295920022655;
                  }
                } else {
                  result[0] += 0.003033285416174097;
                }
              } else {
                result[0] += 0.028790647584350716;
              }
            }
          } else {
            result[0] += -0.020823069092231476;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
          result[0] += -0.012143488339037984;
        } else {
          result[0] += -0.060626225779209834;
        }
      }
    }
  } else {
    result[0] += 0.000984980163657732;
  }
  if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
      result[0] += -2.5984247007392134e-05;
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.015335936277809468;
        } else {
          result[0] += -0.08236685857496967;
        }
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.944137096405030185) ) ) {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0008581199336184054;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.890260934829712802) ) ) {
                    result[0] += -0.03418092934285403;
                  } else {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.024929951763451422;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += -0.01949300083050906;
                      } else {
                        result[0] += 0.04004629852709973;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.03272936727734574;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.31496810913086115) ) ) {
                      result[0] += -0.06504783346635191;
                    } else {
                      result[0] += 0.05110084142047094;
                    }
                  } else {
                    result[0] += 0.025358378962725067;
                  }
                } else {
                  result[0] += 0.03602018898194275;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.166635274887085849) ) ) {
                  result[0] += 0.002941492757429482;
                } else {
                  result[0] += -0.03310779803500489;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.09004387753115797;
              } else {
                result[0] += -0.03649418486222462;
              }
            } else {
              result[0] += 0.06939628379538114;
            }
          }
        } else {
          result[0] += 0.028755164165843734;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.0022401772730761492;
            } else {
              result[0] += 0.015650857482422783;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)31.50000000000000355) ) ) {
                result[0] += -0.06943290085435624;
              } else {
                result[0] += 0.016903352324013157;
              }
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.030405395098670192;
                } else {
                  result[0] += -0.08247684154008507;
                }
              } else {
                result[0] += 6.529309853863085e-05;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.645421981811524326) ) ) {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.006099273075151722;
            } else {
              result[0] += -0.03767923495165995;
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0494521117814645;
              } else {
                result[0] += 0.03182550944360782;
              }
            } else {
              result[0] += -0.057166332340381515;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
          result[0] += 0.024420769103258345;
        } else {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.011952124170696387;
            } else {
              result[0] += -0.0018066255408259622;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
              result[0] += 0.03129010476250953;
            } else {
              result[0] += -0.05778426456158418;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
        if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                result[0] += 0.0031741870127921265;
              } else {
                result[0] += -0.06196365721402312;
              }
            } else {
              result[0] += 0.04147262583635966;
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.05027226003946547;
            } else {
              result[0] += -0.061377221124618;
            }
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.799905776977539951) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)16.50000000000000355) ) ) {
                  result[0] += -0.022085636343172697;
                } else {
                  result[0] += 0.0314020643956144;
                }
              } else {
                result[0] += 0.005947801372569065;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.0006230249248774184;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.07728420250041347;
                  } else {
                    result[0] += 0.016736694893906612;
                  }
                } else {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.09210106432481287;
                  } else {
                    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
                        result[0] += -0.0606542175638242;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.04693307722735385;
                        } else {
                          result[0] += -0.04779223658463162;
                        }
                      }
                    } else {
                      result[0] += 0.01623292575762806;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.603942871093750888) ) ) {
              result[0] += 0.008292483898720317;
            } else {
              result[0] += 0.0350253027818019;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            result[0] += -0.01556487157166759;
          } else {
            result[0] += 0.08392132815913446;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.254884481430054599) ) ) {
            result[0] += 0.047084203843037756;
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.071090809951964;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                      result[0] += 0.049871121668268084;
                    } else {
                      result[0] += -0.03663074180123822;
                    }
                  } else {
                    result[0] += -0.06988703550693062;
                  }
                } else {
                  result[0] += 0.02527145647262177;
                }
              }
            } else {
              result[0] += -0.07011350532362802;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
      result[0] += -6.037987899148826e-05;
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.014464045649099401;
        } else {
          result[0] += -0.08038461972618843;
        }
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                result[0] += -0.02216388553211866;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.016622775949082076;
                } else {
                  result[0] += 0.0030291680143673784;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39368534088134943) ) ) {
                      result[0] += -0.07015715235951232;
                    } else {
                      result[0] += 0.04897758054867845;
                    }
                  } else {
                    result[0] += 0.014884070525913149;
                  }
                } else {
                  result[0] += 0.03152983365149093;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.831219434738160068) ) ) {
                  result[0] += 0.00039613318350274675;
                } else {
                  result[0] += -0.0630931651786461;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.047687357368163485;
            } else {
              result[0] += 0.05125674902597265;
            }
          }
        } else {
          result[0] += 0.025307273511459435;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.00018295671101495423;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.99033999443054288) ) ) {
              result[0] += 0.0022729529220249073;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += -0.00504393950531877;
              } else {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                    result[0] += -0.06683884543280952;
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.1007595338260654;
                    } else {
                      result[0] += 0.05927532622863899;
                    }
                  }
                } else {
                  result[0] += -0.01646863396757192;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.010894474120271981;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.03371282514678412;
              } else {
                result[0] += -0.045960480111957316;
              }
            }
          } else {
            result[0] += -0.06873603820010636;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
          result[0] += 0.023224692967433486;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.011309802921739902;
            } else {
              result[0] += -0.0022117661252190677;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.05571951058373471;
            } else {
              result[0] += -0.04639356030691654;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
        if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
            result[0] += 0.016527690799788188;
          } else {
            result[0] += -0.012700680847679533;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += 0.0579397880340318;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
              result[0] += -0.0004312473759158945;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += -0.018827001872764756;
              } else {
                result[0] += 0.02869048373662029;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.24492526054382413) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0011347191210819578;
                } else {
                  result[0] += 0.02441374008956215;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.08470933867691163;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                      result[0] += -0.11890780222579674;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)121.5000000000000142) ) ) {
                        result[0] += -0.017516505431620002;
                      } else {
                        result[0] += 0.06533037089519136;
                      }
                    }
                  }
                } else {
                  result[0] += -0.09596187312521996;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.06888468281857132;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                      result[0] += 0.0005661317995194144;
                    } else {
                      result[0] += -0.04945244063249346;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.026089082049496917;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
                        result[0] += 0.05384821856583501;
                      } else {
                        result[0] += -0.04680673968926298;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                      result[0] += 0.0423504078917282;
                    } else {
                      result[0] += 0.10967809813543249;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.054146775244831094;
                } else {
                  result[0] += 0.02049813294695757;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += -0.057635304939837075;
              } else {
                result[0] += 0.08829778516013492;
              }
            } else {
              result[0] += 0.002584185541155074;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.00346473523611593;
            } else {
              result[0] += 0.10279481128791912;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.220737934112549716) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                  result[0] += -0.0024322517552553395;
                } else {
                  result[0] += -0.07767696035981923;
                }
              } else {
                result[0] += 0.005936708880462804;
              }
            } else {
              result[0] += -0.04946402668718963;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)256.5000000000000568) ) ) {
    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
        result[0] += -1.568998753635959e-05;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.013877035993532814;
          } else {
            result[0] += -0.0777215164887815;
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.236541748046876776) ) ) {
                  result[0] += -0.008547261673475958;
                } else {
                  result[0] += 0.03990501527352261;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.937313556671143466) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.632002353668214667) ) ) {
                        result[0] += -0.05091198560537672;
                      } else {
                        result[0] += 0.011688504772290277;
                      }
                    } else {
                      result[0] += 0.048232435932527726;
                    }
                  } else {
                    result[0] += 0.02931390454621473;
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.166635274887085849) ) ) {
                    result[0] += 0.0037328199651872093;
                  } else {
                    result[0] += -0.028689148333684213;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.08140981801461394;
              } else {
                result[0] += -0.0172864565367505;
              }
            }
          } else {
            result[0] += 0.02218681628209971;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)17.50000000000000355) ) ) {
          result[0] += -0.0065254410615884;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.874179124832154208) ) ) {
                result[0] += 1.5011460629664456e-05;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                  result[0] += 0.009654500553633304;
                } else {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.030994184302922903;
                  } else {
                    result[0] += -0.08211693430603859;
                  }
                }
              }
            } else {
              result[0] += 0.007728508813005597;
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                  result[0] += 0.004345669108951794;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                      result[0] += 0.03105343372176929;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.777633190155030185) ) ) {
                        result[0] += 0.023462552662786587;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.012083174677625331;
                        } else {
                          result[0] += 0.02237555316840014;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.018684928340856718;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.05806606422987679;
                } else {
                  result[0] += -0.0018785184101806321;
                }
              }
            } else {
              result[0] += 0.024275877383497388;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.09278297424316584) ) ) {
                result[0] += -0.0049087622320249475;
              } else {
                result[0] += -0.06880262786630907;
              }
            } else {
              result[0] += -0.062062090711749285;
            }
          } else {
            result[0] += -0.05108350254041593;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
            result[0] += -0.070337109551735;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.799905776977539951) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.872101783752442294) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0390463986295687;
                  } else {
                    result[0] += 0.00787777766746783;
                  }
                } else {
                  result[0] += 0.06236285930169844;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
                  result[0] += -0.001398147615071318;
                } else {
                  result[0] += -0.06140435096511803;
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.175969600677492011) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                        result[0] += -0.0020543611872156516;
                      } else {
                        result[0] += 0.059555221871101564;
                      }
                    } else {
                      result[0] += -0.025314969150217134;
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)3.02604460716247603) ) ) {
                      result[0] += -0.04649587707210945;
                    } else {
                      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.1395803299441594;
                      } else {
                        result[0] += -0.06586616799755628;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.047493481055982216;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)81.50000000000001421) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                          result[0] += -0.03567040863597396;
                        } else {
                          result[0] += 0.01211003463215961;
                        }
                      } else {
                        result[0] += -0.03744555691342257;
                      }
                    } else {
                      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.01856459331084485;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                            result[0] += 0.024978199752148683;
                          } else {
                            result[0] += 0.11210345297285002;
                          }
                        }
                      } else {
                        result[0] += -0.08514030160975969;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                  result[0] += 0.020202025120752577;
                } else {
                  result[0] += -0.009254792635587107;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
        result[0] += -0.11815884960298828;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.10085550982973267;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.400641441345215288) ) ) {
                  result[0] += 0.10427878024492786;
                } else {
                  result[0] += -0.03813596695511646;
                }
              } else {
                result[0] += 0.02301600952875999;
              }
            } else {
              result[0] += 0.024030799273635137;
            }
          }
        } else {
          result[0] += -0.024124189962161785;
        }
      }
    } else {
      result[0] += -0.05329935745469945;
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.24492526054382413) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)256.5000000000000568) ) ) {
            result[0] += 0.0010156599286764143;
          } else {
            result[0] += -0.02520444728277323;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -0.07695957540468867;
              } else {
                result[0] += -0.014482685138477362;
              }
            } else {
              result[0] += 0.02921359048419027;
            }
          } else {
            result[0] += -0.08258139154225695;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.825982809066773349) ) ) {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += -8.513926537873851e-05;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.56119251251220881) ) ) {
                  result[0] += -0.02106692783969377;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.03295866912589955;
                    } else {
                      result[0] += -0.029275565173504117;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.303973913192749912) ) ) {
                      result[0] += -0.07008773948964048;
                    } else {
                      result[0] += 0.06153780877023443;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.013484079025641674;
                } else {
                  result[0] += -0.007539030363427417;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.975242614746095526) ) ) {
                  result[0] += 0.006996712290505901;
                } else {
                  result[0] += 0.13366300649372484;
                }
              } else {
                result[0] += -0.022673768916442582;
              }
            } else {
              result[0] += 0.0033127727198340954;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.009175135461736829;
            } else {
              result[0] += -0.04199798555657833;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.763591527938843662) ) ) {
                  result[0] += -0.07344713573586419;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                    result[0] += -0.054413827162727926;
                  } else {
                    result[0] += 0.014701298703503228;
                  }
                }
              } else {
                result[0] += -0.09282760689774633;
              }
            } else {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01728905246084647;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02097257004918299;
                } else {
                  result[0] += 0.14175704807823694;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.022538185119630683) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.018568926872687443;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0009702534239398684;
            } else {
              result[0] += -0.03240973842052929;
            }
          } else {
            result[0] += 0.007698766119069819;
          }
        }
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.019355729593705306;
        } else {
          result[0] += 0.06063713018456518;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.008708929472883415;
              } else {
                result[0] += -0.023401182063650525;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += 0.001433266434980807;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08896788511014826;
                  } else {
                    result[0] += -0.025654501494332772;
                  }
                } else {
                  result[0] += 0.055884456451333026;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
              result[0] += -0.06296263411461005;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
                result[0] += -0.02592757160522364;
              } else {
                result[0] += 0.01728120200549568;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.006316375697816997;
                } else {
                  result[0] += 0.013730118227902564;
                }
              } else {
                result[0] += 0.001720521384649727;
              }
            } else {
              result[0] += 0.009534533980564919;
            }
          } else {
            result[0] += 0.020338801561017278;
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)162.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                result[0] += 0.04126587505625537;
              } else {
                result[0] += -0.043353185952808126;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.09305137882180005;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.08813605659596535;
                } else {
                  result[0] += 0.009094707280037062;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.874179124832154208) ) ) {
                result[0] += -0.0003301356856984405;
              } else {
                result[0] += -0.016047913729763368;
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.00012359452559522116;
              } else {
                result[0] += -0.06419331125696902;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.722943305969239169) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03854378162395467;
            } else {
              result[0] += -0.0014772510338341266;
            }
          } else {
            result[0] += -0.04484712469155486;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
            result[0] += 0.003430285129286902;
          } else {
            result[0] += -0.062013060357693744;
          }
        } else {
          result[0] += 0.015791108198832386;
        }
      } else {
        result[0] += 0.0694980796501689;
      }
    }
  }
  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
      result[0] += 0.07405812575393152;
    } else {
      result[0] += -0.1330220762940033;
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
            result[0] += -0.04910102032566329;
          } else {
            result[0] += -0.01363210919008953;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
            result[0] += -0.04972677419959842;
          } else {
            result[0] += 0.0030186293739592252;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.0006568971406631104;
        } else {
          result[0] += 0.06795733807006742;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.24492526054382413) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0009891799918182087;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
                    result[0] += -0.012786845699372669;
                  } else {
                    result[0] += -0.07303964734707348;
                  }
                } else {
                  result[0] += 0.02662581942076165;
                }
              } else {
                result[0] += -0.07957402855824268;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.0008498841924255908;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.019237021407735085;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0015576751171056714;
                    } else {
                      result[0] += -0.02093309371715384;
                    }
                  }
                }
              } else {
                result[0] += 0.0018750163376308688;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.029861007325464452;
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.05008636293699118;
                  } else {
                    result[0] += -0.016419310306576267;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.609406948089601386) ) ) {
                    result[0] += -0.02821453595583788;
                  } else {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.019262632905733143;
                    } else {
                      result[0] += 0.13955591401616835;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.020540529379419345;
          } else {
            result[0] += -0.004658955281240972;
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.222574234008789951) ) ) {
                result[0] += 0.004198968484617173;
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.901921629905701128) ) ) {
                    result[0] += 0.049254627822347935;
                  } else {
                    result[0] += -0.04727019648178582;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                    result[0] += 0.0006002077720098473;
                  } else {
                    result[0] += -0.018942798846238833;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                      result[0] += 0.028766455282441324;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.968900680541993964) ) ) {
                          result[0] += 0.01869274856756404;
                        } else {
                          result[0] += -0.0033983892486144558;
                        }
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.068990230560303623) ) ) {
                          result[0] += -0.030981497614501925;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.700598716735840066) ) ) {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.04688412800836228;
                            } else {
                              result[0] += -0.03142287549250835;
                            }
                          } else {
                            result[0] += -0.010058647435124492;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.041269512805317037;
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.005715885240550079;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
                      result[0] += -0.012713062390736813;
                    } else {
                      result[0] += -0.07081586569990982;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)216.5000000000000284) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += -0.09892619778750035;
                      } else {
                        result[0] += -0.004022428737601891;
                      }
                    } else {
                      result[0] += -0.028098492667127953;
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.278613805770874912) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
                          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.009580408114051704;
                          } else {
                            result[0] += -0.03644485538883462;
                          }
                        } else {
                          result[0] += 0.059976528824310066;
                        }
                      } else {
                        result[0] += 0.02858955834757525;
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
                            result[0] += 0.03088789421244888;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.997585535049439365) ) ) {
                              result[0] += 0.005521673623422859;
                            } else {
                              result[0] += -0.09811129078098883;
                            }
                          }
                        } else {
                          result[0] += 0.07428374696400457;
                        }
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
                            result[0] += 0.005580417473833059;
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
                              result[0] += -0.06130341148391656;
                            } else {
                              result[0] += 0.08736176212672135;
                            }
                          }
                        } else {
                          result[0] += 0.04933708280644214;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.04446902057162487;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.010625892908666515;
            } else {
              result[0] += 3.528725014588214e-05;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
            result[0] += -0.15023574380130167;
          } else {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.048201330898558256;
            } else {
              result[0] += -0.007119200238241634;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)186.5000000000000284) ) ) {
            result[0] += -0.003890315223893175;
          } else {
            result[0] += 0.008376463032857917;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
            result[0] += -0.007286481367775915;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.016676603873096194;
            } else {
              result[0] += -0.06823177258272592;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.001548339128811092;
          } else {
            result[0] += -0.04131950118167797;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.022538185119630683) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.04242737759907045;
            } else {
              result[0] += -0.015396857815886731;
            }
          } else {
            result[0] += 0.045564728523209044;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05081367492675959) ) ) {
            result[0] += -0.003522048675336653;
          } else {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
                result[0] += -0.02236124388773442;
              } else {
                result[0] += 0.03504391529577932;
              }
            } else {
              result[0] += 0.07201568653370219;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.10220111781006243;
              } else {
                result[0] += 0.019326644875375446;
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.03188889425325981;
              } else {
                result[0] += -0.0019353252040052544;
              }
            }
          } else {
            result[0] += -0.0373114187910346;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += 0.01681181096851887;
            } else {
              result[0] += -0.01592885101270992;
            }
          } else {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.04341159525719576;
            } else {
              if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.09186679539208639;
              } else {
                result[0] += -0.044933769631459675;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.13839721679687678) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)28.50000000000000355) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)26.50000000000000355) ) ) {
                      result[0] += -0.00627391228267531;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                        result[0] += 0.03715221979722172;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                            result[0] += 0.0055337602229933;
                          } else {
                            result[0] += -0.15750501920050758;
                          }
                        } else {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.95229363441467374) ) ) {
                            result[0] += -0.059015861536770936;
                          } else {
                            result[0] += 0.0181969056573035;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.0019049710433813338;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)117.5000000000000142) ) ) {
                      result[0] += -0.023394896623579756;
                    } else {
                      result[0] += 0.01088601967883357;
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
                      result[0] += 0.04387255782240848;
                    } else {
                      result[0] += 0.009836656718505777;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                      result[0] += 0.024782771517407966;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                        result[0] += 0.027215608824546014;
                      } else {
                        result[0] += -0.03032922969968091;
                      }
                    }
                  } else {
                    result[0] += -0.06071502136517587;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.019353342219670683;
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.02232040294932515;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.02573886896021121;
                        } else {
                          result[0] += -0.015273434358751906;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.017991671008130573;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0010274192279821433;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.020033601762995373;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                    result[0] += -0.051494107949066815;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.010309908776906343;
                    } else {
                      result[0] += -0.05552453450237234;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.009502598130812195;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.028551670074184805;
                } else {
                  result[0] += 0.011827247435005477;
                }
              } else {
                result[0] += 0.05340052569247287;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)159.5000000000000284) ) ) {
        result[0] += 0.0014344374757090054;
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.017584248800142806;
        } else {
          result[0] += -0.008162414207756704;
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
          result[0] += 0.00024830278932039994;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += 0.0009904577247172549;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.043268828316842534;
            } else {
              result[0] += -0.007224712093591837;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
            result[0] += 0.015358791065918302;
          } else {
            result[0] += -0.06902930202917872;
          }
        } else {
          result[0] += -0.018580457061439563;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)186.5000000000000284) ) ) {
            result[0] += -0.003597424877377851;
          } else {
            result[0] += 0.007957116084077839;
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.758822202682496005) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.004422465369161529;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.358708143234253818) ) ) {
                result[0] += -0.015697556385509442;
              } else {
                result[0] += 0.06156750724181509;
              }
            }
          } else {
            result[0] += -0.04610053821985732;
          }
        }
      } else {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.013364020191424643;
        } else {
          result[0] += -0.06588529140976966;
        }
      }
    } else {
      if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.05369597389755629;
            } else {
              result[0] += 0.0033054774306171043;
            }
          } else {
            result[0] += 0.001512027436206306;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += -0.037941136685070584;
            } else {
              result[0] += 0.0033527874562113806;
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.065660476684572089) ) ) {
                result[0] += 0.005584802098093177;
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.01599344265564727;
                } else {
                  result[0] += -0.04201920931656397;
                }
              }
            } else {
              result[0] += -0.005202177753134335;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
              result[0] += -0.018971842556997954;
            } else {
              result[0] += -0.11678250591827477;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)28.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
                result[0] += 0.002127133917174841;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)19.50000000000000355) ) ) {
                  result[0] += -0.0052641249547896105;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
                    result[0] += -0.08046395446188431;
                  } else {
                    result[0] += -0.027474999092476277;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.89399480819702326) ) ) {
                  result[0] += -0.006884628530705867;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += 0.03380261934281793;
                  } else {
                    result[0] += -0.046375564131380366;
                  }
                }
              } else {
                result[0] += -0.015352214821194924;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.853236675262452948) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                  result[0] += 0.005705378672375298;
                } else {
                  result[0] += -0.044426083927207316;
                }
              } else {
                result[0] += 0.01249629254942173;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += 0.021409277326713883;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.0033040742158540123;
                  } else {
                    result[0] += -0.046602461492295244;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.869292974472046787) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.01833335749373121;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.905434608459474433) ) ) {
                        result[0] += -0.024606345888170827;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.02546706240238615;
                        } else {
                          result[0] += -0.026746195748088165;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.601027011871338779) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.017151061956184918;
                      } else {
                        result[0] += 0.030116408436498893;
                      }
                    } else {
                      result[0] += -0.26055230852817113;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.022391141322442493;
                    } else {
                      result[0] += -0.029267947071685355;
                    }
                  } else {
                    result[0] += 0.03705254288386397;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
                    result[0] += -0.07417248513741774;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
                      result[0] += -0.03849816114282558;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                          result[0] += -0.03183806250694129;
                        } else {
                          result[0] += 0.01755134314554027;
                        }
                      } else {
                        result[0] += -0.1157505262187903;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                    result[0] += -0.02204026783937518;
                  } else {
                    result[0] += -0.12062662368460873;
                  }
                }
              } else {
                result[0] += 0.009284667693014515;
              }
            } else {
              result[0] += 0.007703699388966163;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)37.50000000000000711) ) ) {
        result[0] += -0.0008111103647851231;
      } else {
        result[0] += 0.002923725688506084;
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.396947860717774326) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.109245061874390537) ) ) {
              result[0] += -0.020086040950373585;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.01412338170549985;
              } else {
                result[0] += -0.0068973643494070385;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
              result[0] += -0.0040330843908639264;
            } else {
              result[0] += -0.038291956955737144;
            }
          }
        } else {
          result[0] += 0.004255802538570517;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
          result[0] += -0.05691831729899456;
        } else {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.06653997555751894;
          } else {
            result[0] += -0.0071035155351064005;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[33].missing != -1) && (data[33].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 0.06822491534697248;
      } else {
        result[0] += -0.08843013319637606;
      }
    } else {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06276416778564631) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
            result[0] += 0.013362976044586979;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.002892439868529959;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.109245061874390537) ) ) {
                  result[0] += 0.025445333320786218;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.386624813079835761) ) ) {
                    result[0] += 0.007478743725525875;
                  } else {
                    result[0] += -0.03709920116353055;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.004493485753597827;
                } else {
                  result[0] += 0.01692674072729419;
                }
              } else {
                result[0] += -0.002972536608115339;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += -0.055932237250409425;
                } else {
                  result[0] += 0.009132807637540647;
                }
              } else {
                result[0] += -0.04608259554601196;
              }
            } else {
              result[0] += -0.08483639392451214;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.05104221486475357;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.01048520448032554;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.06797530536156259;
                    } else {
                      result[0] += -0.004212996537913674;
                    }
                  }
                }
              } else {
                result[0] += 0.030497214836734694;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0037409170457339625;
              } else {
                result[0] += -0.02713922638873823;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.970085620880127397) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.004022594768971363;
          } else {
            result[0] += -0.005244419723510905;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05081367492675959) ) ) {
                result[0] += -0.001747266376851358;
              } else {
                result[0] += 0.037444621234141884;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += 0.0026016889762679484;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                  result[0] += -0.01652097210131907;
                } else {
                  result[0] += -0.0578043154570728;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.009853658422774816;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0247175688214356;
                    } else {
                      result[0] += 0.006335790259043433;
                    }
                  } else {
                    result[0] += -0.0008040744663528401;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += 0.006526037030503182;
                } else {
                  result[0] += -0.016402756854481542;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.89399480819702326) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.010353337040765041;
                } else {
                  result[0] += -0.0009085425521335143;
                }
              } else {
                if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.020241511599987425;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += 0.01133348136393187;
                  } else {
                    result[0] += -0.046811497937321644;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.940192461013794833) ) ) {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0009759442782449311;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
            result[0] += -0.000499923222447629;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.2829140969194523;
                } else {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.022154577651810153;
                  } else {
                    result[0] += 0.022904874012862283;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.060753152700339985;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.473832368850708896) ) ) {
                        result[0] += -0.06635673508289257;
                      } else {
                        result[0] += 0.10705398673542993;
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.004888079716419492;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                          result[0] += -0.06814221044603196;
                        } else {
                          result[0] += -0.007807825170239005;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.03808203190294503;
                      } else {
                        result[0] += 0.0003121805335394969;
                      }
                    } else {
                      result[0] += -0.0880133202511269;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += -0.07571965385766086;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)103.5000000000000142) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.61020660400390803) ) ) {
                      result[0] += -0.03895909219035304;
                    } else {
                      result[0] += 0.14513006606891024;
                    }
                  } else {
                    result[0] += -0.021989850709572163;
                  }
                } else {
                  result[0] += -0.0720993154634862;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.0659595812124064;
        } else {
          result[0] += -0.01001677318081777;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
        result[0] += -0.06629007246538214;
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)240.5000000000000284) ) ) {
          result[0] += -0.02188699709977995;
        } else {
          result[0] += 0.02958726249711706;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.003470696592649194;
          } else {
            result[0] += 0.0073708671603656215;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
            result[0] += -0.006802701596909386;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.017195812181944475;
            } else {
              result[0] += -0.06313140432381927;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.05572467388077055;
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0040932125423995816;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.022538185119630683) ) ) {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.029441232837506572;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += 0.07244395967865784;
                  } else {
                    result[0] += -0.013500321692956486;
                  }
                }
              } else {
                result[0] += 0.04334281879430681;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)9.167253971099855292) ) ) {
              result[0] += -0.06401221380353332;
            } else {
              result[0] += 0.15909679891265519;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.03260610954087439;
        } else {
          result[0] += -0.0010445570093640317;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.0034982488001698963;
            } else {
              result[0] += -0.18369858891668028;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
              result[0] += -0.10667929465120021;
            } else {
              result[0] += -0.008076774325321405;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.874179124832154208) ) ) {
                    result[0] += -0.019838811053374945;
                  } else {
                    result[0] += -0.08756936516802946;
                  }
                } else {
                  result[0] += 0.014686939419625298;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.27828097343444913) ) ) {
                  result[0] += 0.024819403831613664;
                } else {
                  result[0] += -0.006361340423737458;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                  result[0] += 0.020489500130168124;
                } else {
                  result[0] += -0.10774073964137164;
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.05060957810142615;
                        } else {
                          result[0] += -0.019578732118197988;
                        }
                      } else {
                        result[0] += 0.0026364201198471647;
                      }
                    } else {
                      result[0] += -0.0636997292246105;
                    }
                  } else {
                    result[0] += -0.031019297330778774;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.934722661972046787) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1479225158691424) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.0024496856118738693;
                      } else {
                        result[0] += -0.07516671553785173;
                      }
                    } else {
                      result[0] += 0.025336816571552814;
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.023846818552961436;
                    } else {
                      result[0] += 0.03810990115929183;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.45601701736450373) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                result[0] += -0.0316358498149202;
              } else {
                result[0] += 0.005868397469953595;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.008754958180280995;
              } else {
                result[0] += -0.07347881354293063;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.09278297424316584) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.0017379680900009133;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.183107137680054599) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                  result[0] += -0.051431771388617466;
                } else {
                  result[0] += -0.007267139511512451;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += 0.012014097282319933;
                } else {
                  result[0] += -0.056730419575463556;
                }
              }
            } else {
              result[0] += -0.0006841637227351284;
            }
          } else {
            result[0] += 0.008633418560024643;
          }
        }
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.547126770019532138) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += 0.019962180389167358;
                } else {
                  result[0] += -0.004386259859602135;
                }
              } else {
                result[0] += 0.042720174399274835;
              }
            } else {
              result[0] += 0.07650650523681655;
            }
          } else {
            result[0] += 0.05468982943671102;
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.005622150323673266;
          } else {
            result[0] += -0.07970385303326402;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
            result[0] += -0.008934553152602432;
          } else {
            result[0] += -0.09330197206868443;
          }
        } else {
          result[0] += -0.01728295313519871;
        }
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.042324036133251766;
            } else {
              result[0] += 0.008586587149582622;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
              result[0] += -0.057177268455413745;
            } else {
              result[0] += -0.005685510856535098;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
            result[0] += -0.005342837351548462;
          } else {
            result[0] += -0.0734744318602962;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.00047452613840280643;
  } else {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.940192461013794833) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.645740747451783115) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
          result[0] += -0.005439954636128377;
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.89399480819702326) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.64700984954834162) ) ) {
                    result[0] += 0.0037770348111155305;
                  } else {
                    result[0] += 0.03034753859277586;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.007150030565138923;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.006892942873263321;
                        } else {
                          result[0] += 0.13779455084547443;
                        }
                      } else {
                        result[0] += 0.00967587807903898;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.007724742491056556;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                        result[0] += -0.023133369667558586;
                      } else {
                        result[0] += -0.07061459993805;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59605169296264826) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.10504861090479432;
                          } else {
                            result[0] += -0.0051240136918840875;
                          }
                        } else {
                          result[0] += 0.03439571837844831;
                        }
                      } else {
                        result[0] += -0.01771669240688449;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.062072674658065444;
                      } else {
                        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
                            result[0] += 0.11476674196547959;
                          } else {
                            result[0] += -0.09830503335189052;
                          }
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
                            result[0] += 0.001963253498452806;
                          } else {
                            result[0] += 0.11611080565130633;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                      result[0] += -0.05876755528457649;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59605169296264826) ) ) {
                        result[0] += 0.002491029995149866;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.05986218138941889;
                        } else {
                          result[0] += -0.0025352498835012082;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.48638534545898615) ) ) {
                      result[0] += -0.0027392088699823946;
                    } else {
                      result[0] += -0.05693358366940772;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.98246049880981623) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.037518621580985886;
                      } else {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.05546836889736855;
                        } else {
                          result[0] += -0.12057634158490392;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.013251020790685755;
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.014093571212593524;
                        } else {
                          result[0] += 0.08820889105478173;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0884132385253924) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.607985973358155185) ) ) {
                    result[0] += -0.08840412356748;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.499747991561890537) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.08143634945633993;
                      } else {
                        result[0] += -0.01888887635126235;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.763591527938843662) ) ) {
                        result[0] += 0.008038749747940692;
                      } else {
                        result[0] += -0.029619466166993735;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.607985973358155185) ) ) {
                    result[0] += 0.07301969531488851;
                  } else {
                    result[0] += 0.008729306925531064;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.799905776977539951) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59605169296264826) ) ) {
                    result[0] += -0.03814118525950134;
                  } else {
                    result[0] += -0.09307933863860582;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59605169296264826) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.524927973747253862) ) ) {
                      result[0] += -0.01299959337578199;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                        result[0] += -0.0415376686729235;
                      } else {
                        result[0] += 0.09424344806691724;
                      }
                    }
                  } else {
                    result[0] += -0.03887154608575931;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47078418731689631) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.1289354525910855;
                  } else {
                    result[0] += 0.012272227722500546;
                  }
                } else {
                  result[0] += -0.01811448301319866;
                }
              } else {
                result[0] += -0.07836075761780611;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47078418731689631) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.924915313720704901) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
                      result[0] += -0.039618443084319326;
                    } else {
                      result[0] += 0.004176415191232946;
                    }
                  } else {
                    result[0] += 0.013612398851526795;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06276416778564631) ) ) {
                    result[0] += -0.069817794601977;
                  } else {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.01410846621751438;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.524927973747253862) ) ) {
                        result[0] += 0.07248133180931297;
                      } else {
                        result[0] += -0.11828365408436754;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                  result[0] += 0.08477398472560144;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                    result[0] += -0.039569383913047394;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)3.164860486984253374) ) ) {
                      result[0] += 0.043607601116862994;
                    } else {
                      result[0] += -0.195169700033789;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.007927760933673764;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
        result[0] += -0.06401564161742866;
      } else {
        result[0] += -0.015872096037429503;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
        result[0] += -0.04797854938609786;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.88174462318420499) ) ) {
              result[0] += -0.025385055719823232;
            } else {
              result[0] += -0.008287354997069084;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.040467519645599914;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.012103498294168723;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                  result[0] += -0.017780109126086904;
                } else {
                  result[0] += 0.01561798226042059;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
            result[0] += -0.0008343155302430095;
          } else {
            result[0] += 0.06482659323482655;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.0006270318536702037;
        } else {
          if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)113.5000000000000142) ) ) {
                  result[0] += -0.008235137676501655;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                    result[0] += -0.010759258553772329;
                  } else {
                    result[0] += 0.02526679693834076;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.059244811450407775;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.08644587101429661;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
                          result[0] += -0.0076612914029488225;
                        } else {
                          result[0] += -0.0450749196785994;
                        }
                      } else {
                        result[0] += -0.0639092489405094;
                      }
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.014281198333327522;
                        } else {
                          result[0] += -0.045977820876949536;
                        }
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.04654110656256796;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)121.5000000000000142) ) ) {
                              result[0] += 0.11146514541341093;
                            } else {
                              result[0] += 0.03192073194304492;
                            }
                          } else {
                            result[0] += -0.08678600900483568;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.009795392901178104;
              } else {
                result[0] += 0.06050954087402221;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.08553763433807307;
            } else {
              result[0] += -0.023476916223641484;
            }
          }
        }
      } else {
        result[0] += -0.08082427916143728;
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.109050035476685458) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.182021141052246982) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += 0.0018638767029576472;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.890260934829712802) ) ) {
                  result[0] += -0.011865744982400266;
                } else {
                  result[0] += 0.06588959048155586;
                }
              }
            } else {
              result[0] += -0.0651312011763206;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)103.5000000000000142) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.296216011047365058) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.090008378028870073) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.05334038226536351;
                  } else {
                    result[0] += 0.03299535092905394;
                  }
                } else {
                  result[0] += -0.0506633761010713;
                }
              } else {
                result[0] += 0.022291030623578578;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.210062026977539951) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.937313556671143466) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.418317794799805576) ) ) {
                          result[0] += 0.003546192148486525;
                        } else {
                          result[0] += 0.046820215028000094;
                        }
                      } else {
                        result[0] += -0.006138819040955125;
                      }
                    } else {
                      result[0] += -0.05078211110133121;
                    }
                  } else {
                    result[0] += -0.009119450567102836;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.08524750027555637;
                  } else {
                    result[0] += 0.0063572955992845185;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += -0.058217474814964365;
                  } else {
                    result[0] += 0.03205397729806555;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.329718828201294833) ) ) {
                    result[0] += 0.11277471492765112;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.340511322021485263) ) ) {
                        result[0] += -0.11173837390319603;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                          result[0] += 0.055919694587760777;
                        } else {
                          result[0] += -0.02342347770038063;
                        }
                      }
                    } else {
                      result[0] += 0.07092525913119199;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.0016621270503077877;
          } else {
            result[0] += 0.044251535264346026;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
          result[0] += -0.0025110944992000252;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.590443611145021308) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                  result[0] += 0.0134991690667498;
                } else {
                  result[0] += -0.06905064007369115;
                }
              } else {
                result[0] += -0.06663682243675423;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.132412433624269354) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.002529837848017084;
                } else {
                  result[0] += -0.07396218247599516;
                }
              } else {
                result[0] += 0.004680312992758955;
              }
            }
          } else {
            result[0] += -0.02974874025067613;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
        result[0] += -0.03875632451159229;
      } else {
        result[0] += -0.012350308690313817;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
        result[0] += -0.04404304316567124;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.06681112556276768;
              } else {
                result[0] += 0.09905126712006476;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03690209931817344;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
                    result[0] += -0.031833157382278106;
                  } else {
                    result[0] += 0.0034316782310175884;
                  }
                }
              } else {
                result[0] += -0.0028167927063289504;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.036547020971081826;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.010119492969230589;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                  result[0] += -0.016014618015526588;
                } else {
                  result[0] += 0.014878133978223716;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
            result[0] += -0.0007099510240956026;
          } else {
            result[0] += 0.0577393635881516;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)7.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)12.50000000000000178) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.68180561065674006) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += -0.021300696353733994;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.296216011047365058) ) ) {
                  result[0] += -0.026279948000966575;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.03169349926826762;
                  } else {
                    result[0] += -0.002301144355130416;
                  }
                }
              } else {
                result[0] += 0.010715954781374153;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.058524850781846355;
              } else {
                result[0] += 0.09698232516132907;
              }
            } else {
              result[0] += -0.019428010704712194;
            }
          }
        } else {
          result[0] += 0.00036332295624920467;
        }
      } else {
        result[0] += -0.0781895956099937;
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.109050035476685458) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.156774044036865678) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)19.50000000000000355) ) ) {
                  result[0] += -0.05957418033527115;
                } else {
                  result[0] += -0.0028004092325246643;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.0076702415394095036;
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.07108016343082067;
                    } else {
                      result[0] += -0.019390010901612634;
                    }
                  } else {
                    result[0] += 0.040027510165001216;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.04635410542322291;
                  } else {
                    result[0] += 0.00552469345267034;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)203.5000000000000284) ) ) {
                    result[0] += 0.007847082223982288;
                  } else {
                    result[0] += -0.06730170968560781;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.725620865821838823) ) ) {
                  result[0] += -0.05929107276428969;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.04037270974676251;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
                      result[0] += -0.01857797083240138;
                    } else {
                      result[0] += 0.02053914767669582;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.03408523433697707;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.003480691391775532;
              } else {
                result[0] += 0.21546064140097398;
              }
            }
          }
        } else {
          result[0] += 0.00274062142670092;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
          result[0] += -0.0025175410800095223;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.98246049880981623) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
              result[0] += 0.0011307609532928687;
            } else {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += -0.036706656357465284;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
                    if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.132412433624269354) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
                          result[0] += 0.02272919888854032;
                        } else {
                          result[0] += -0.11637227981862991;
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.103847122513058;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1479225158691424) ) ) {
                            result[0] += 0.0595926689545942;
                          } else {
                            result[0] += -0.10484559990601339;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.265274047851563388) ) ) {
                        result[0] += -0.00611431687573652;
                      } else {
                        result[0] += 0.029842443874401187;
                      }
                    }
                  } else {
                    result[0] += -0.037648559229823415;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
                  result[0] += -0.0842131093211722;
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.51615929603576749) ) ) {
                    result[0] += 0.0022804753856502462;
                  } else {
                    result[0] += -0.05107683926524732;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)212.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)19.50000000000000355) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0761816858936481;
                } else {
                  result[0] += 0.0868252202342083;
                }
              } else {
                result[0] += -0.03185473422196026;
              }
            } else {
              result[0] += 0.03493658871382173;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.260092735290528232) ) ) {
        result[0] += -0.06107905356083512;
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)227.5000000000000284) ) ) {
          result[0] += -0.021250497528327392;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
            result[0] += 0.025588670105608502;
          } else {
            result[0] += -0.10939778211431878;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)7.500000000000000888) ) ) {
      result[0] += 0.0003581235874250835;
    } else {
      result[0] += -0.07557985708678941;
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.109050035476685458) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.463993549346925604) ) ) {
                  result[0] += -0.010820000605879114;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.046835988691395235;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.030113297816448444;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9916415214538592) ) ) {
                          result[0] += 0.146485449401607;
                        } else {
                          result[0] += -0.024998971923240982;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                          result[0] += 0.015725752651807152;
                        } else {
                          result[0] += 0.07244468272981082;
                        }
                      } else {
                        result[0] += -0.008600604343349739;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0513344648490762;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.222574234008789951) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += 0.03081529356149862;
                    } else {
                      result[0] += -2.6451217518722314e-05;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.238486170768738237) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.238486170768738237) ) ) {
                          result[0] += 0.09461227835591696;
                        } else {
                          result[0] += 0.02064440107052;
                        }
                      } else {
                        result[0] += -0.021589591777634656;
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.64700984954834162) ) ) {
                        if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.024032064814964997;
                        } else {
                          result[0] += -0.002768623823629593;
                        }
                      } else {
                        result[0] += 0.0557838672510329;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.05081367492675959) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.12985633979011582;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.06817054748535334) ) ) {
                            result[0] += -0.0067240058329162385;
                          } else {
                            result[0] += 0.03433977914429695;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.569529533386231357) ) ) {
                            result[0] += -0.09207179597357538;
                          } else {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += -0.019103782665115934;
                            } else {
                              result[0] += -0.07883909567555629;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.07934224794846634;
                          } else {
                            result[0] += 0.03809300760012452;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.008891460560060715;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                          result[0] += -0.012845125977787567;
                        } else {
                          result[0] += 0.054195742920254975;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.53813362121582209) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.297559976577759233) ) ) {
                        result[0] += -0.06508121733439469;
                      } else {
                        result[0] += 0.0016169006018742643;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.07157460156216082;
                        } else {
                          result[0] += -0.002161824198984068;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                          result[0] += -0.10141174051522882;
                        } else {
                          result[0] += -0.003996841760786437;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.023116079736307166;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.276966691017151323) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
                  result[0] += 0.004417958050189834;
                } else {
                  result[0] += -0.09142320764677969;
                }
              } else {
                result[0] += -0.036362347327181514;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07031325831758593;
                    } else {
                      result[0] += -0.0022466520829210693;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                      result[0] += 0.010361080156522769;
                    } else {
                      result[0] += 0.039364025547417414;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.418317794799805576) ) ) {
                        result[0] += 0.025535941184008305;
                      } else {
                        result[0] += -0.03529089767405491;
                      }
                    } else {
                      result[0] += -0.0881226086551737;
                    }
                  } else {
                    result[0] += -0.00022634444540932903;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.34467267990112482) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.010609724068562157;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
                      result[0] += -0.08344005506929464;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                        result[0] += -0.06311348227166702;
                      } else {
                        result[0] += -0.004543347227844575;
                      }
                    }
                  }
                } else {
                  result[0] += 0.0024091813560909066;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.028962824733610844;
                } else {
                  result[0] += -0.0758949640463889;
                }
              } else {
                result[0] += 0.048530001360550364;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                result[0] += -0.04230834381305143;
              } else {
                result[0] += -0.10474837070627317;
              }
            }
          } else {
            result[0] += -0.0015956296039335235;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
          result[0] += -0.002519346078272245;
        } else {
          result[0] += -0.015779524833865976;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.254884481430054599) ) ) {
        result[0] += -0.06257583119233932;
      } else {
        result[0] += -0.015355523796277299;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.00032999126771688037;
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.109050035476685458) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)42.50000000000000711) ) ) {
                  result[0] += -0.13101283765830743;
                } else {
                  result[0] += 0.03913290631386723;
                }
              } else {
                result[0] += -0.06974982647588547;
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                  result[0] += 0.027214800805124884;
                } else {
                  result[0] += -0.08888071960814059;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.238486170768738237) ) ) {
                  result[0] += -0.05139936510761587;
                } else {
                  result[0] += 0.057603796654654676;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)212.5000000000000284) ) ) {
                result[0] += 0.04544226833788894;
              } else {
                result[0] += -0.024511058143466005;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
                result[0] += 0.0037351593880473456;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)203.5000000000000284) ) ) {
                  result[0] += -0.09081811012519343;
                } else {
                  result[0] += 0.016936748648587307;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.008969283652614672;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.004881381988526279) ) ) {
                          result[0] += 0.006515388663709725;
                        } else {
                          result[0] += -0.09131547779164068;
                        }
                      } else {
                        result[0] += 0.06478196476418309;
                      }
                    } else {
                      result[0] += -0.07138168331193329;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
                      result[0] += 0.09222511242764031;
                    } else {
                      result[0] += -0.00710762983796415;
                    }
                  }
                }
              } else {
                result[0] += 0.009117785480709024;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.386624813079835761) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
                        result[0] += 0.054180116819181334;
                      } else {
                        result[0] += -0.07625094388969761;
                      }
                    } else {
                      result[0] += 0.067227404732817;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                          result[0] += -0.0718044285619548;
                        } else {
                          result[0] += 0.020187801583339923;
                        }
                      } else {
                        result[0] += 0.050094552503858004;
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.874179124832154208) ) ) {
                        result[0] += -0.023719615101055143;
                      } else {
                        result[0] += -0.0933762633757524;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.238486170768738237) ) ) {
                    result[0] += 0.0338325161025256;
                  } else {
                    result[0] += -0.012152238453128803;
                  }
                }
              } else {
                result[0] += 0.0031812203569025646;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.276966691017151323) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)117.5000000000000142) ) ) {
                result[0] += -0.07085275333121765;
              } else {
                result[0] += -0.01838975525073324;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.524927973747253862) ) ) {
                result[0] += -0.018763167513468633;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
                    result[0] += 0.08462219180389259;
                  } else {
                    result[0] += 0.008646355975369885;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)42.50000000000000711) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
                          result[0] += -0.026585932131143727;
                        } else {
                          result[0] += 0.009945496702015492;
                        }
                      } else {
                        result[0] += 0.02615012121781425;
                      }
                    } else {
                      result[0] += -0.0013249692732982604;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.31479072570800959) ) ) {
                      result[0] += -0.022720333840416745;
                    } else {
                      result[0] += 0.002080089838501833;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.001507262153311168;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
              result[0] += -0.027808011323419847;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.03999293584245925;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.020232744099349012;
                  } else {
                    result[0] += 0.021810978353064045;
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                      result[0] += -0.05952925778844242;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.00892901472501478;
                        } else {
                          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                              result[0] += -0.039927727795965644;
                            } else {
                              result[0] += 0.0793952085906166;
                            }
                          } else {
                            result[0] += -0.052782456295580475;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                          result[0] += -0.06591072664432994;
                        } else {
                          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.08516914498284751;
                          } else {
                            result[0] += 0.006412834442130311;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.003883576663298033;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.07077905814100467;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                result[0] += -0.013496484253388334;
              } else {
                result[0] += -0.06495766641078227;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += -0.026015208049705554;
        } else {
          result[0] += -0.09191603199691821;
        }
      } else {
        result[0] += -0.0077443048513938535;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.24492526054382413) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
        result[0] += 0.0007763410551552496;
      } else {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
            result[0] += -0.002022312099561424;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.002676553487811475;
              } else {
                result[0] += -0.020365576245145203;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += -0.017013994722551194;
              } else {
                result[0] += -0.04936027257084656;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0066661716213822185;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                  result[0] += -0.0068057147013204896;
                } else {
                  result[0] += -0.06282357506343622;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                  result[0] += -0.01634910312387032;
                } else {
                  result[0] += 0.03906891243081739;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.019550430367166333;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.824383735656740058) ) ) {
                  result[0] += -0.04112006386409979;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.01445333957519286;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                          result[0] += -0.09066089028152997;
                        } else {
                          result[0] += 0.008843498740737754;
                        }
                      } else {
                        result[0] += -0.0893782799298867;
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.013750439441409744;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.254884481430054599) ) ) {
                          result[0] += -0.09855199608785621;
                        } else {
                          result[0] += 0.08909747448834487;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.810120582580567294) ) ) {
                    result[0] += -0.029669875802072948;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.905434608459474433) ) ) {
                      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.854362010955811435) ) ) {
                          result[0] += -0.020013428932021946;
                        } else {
                          result[0] += 0.03425645102915106;
                        }
                      } else {
                        result[0] += 0.05819965008786152;
                      }
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.022928308934342244;
                        } else {
                          result[0] += 0.0233791513153546;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.222574234008789951) ) ) {
                          result[0] += 0.14479339784049064;
                        } else {
                          result[0] += 0.03458071660526778;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
                    result[0] += -0.05009394067226752;
                  } else {
                    result[0] += 0.00864390072085782;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.02012051585863978;
      } else {
        result[0] += -0.004414221024389735;
      }
    }
  } else {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.0045037109209331875;
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.665046453475953037) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.0036415108012157883;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.940192461013794833) ) ) {
                        result[0] += -0.015531967490557887;
                      } else {
                        result[0] += -0.0005699859084839943;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.007348932302383972;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.666320323944092685) ) ) {
                        result[0] += -0.05169304682323811;
                      } else {
                        result[0] += -0.018974109472794667;
                      }
                    }
                  }
                } else {
                  result[0] += 0.017842989584637908;
                }
              } else {
                result[0] += 0.0010437683210982932;
              }
            } else {
              result[0] += 0.008402755875949151;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                result[0] += -0.020423584756438486;
              } else {
                result[0] += 0.0037848729494628116;
              }
            } else {
              result[0] += 0.013322633904889589;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
              result[0] += 0.02342887329135115;
            } else {
              result[0] += -0.07445335994483342;
            }
          } else {
            result[0] += 0.040363509371948623;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
            result[0] += -0.0008812703503886035;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.183107137680054599) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.009757815250625503;
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.057080231291701145;
                  } else {
                    result[0] += -0.023994063183675134;
                  }
                }
              } else {
                result[0] += 0.016044303960683768;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                result[0] += -0.02320543333360809;
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                    result[0] += 0.039789100694375405;
                  } else {
                    result[0] += -0.05120887060226753;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.011793811838395648;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.01684600468970852;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                        result[0] += -0.04246462718882217;
                      } else {
                        result[0] += 0.04878861265036316;
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
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.00948229289399166;
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
          result[0] += 0.021157362060455886;
        } else {
          result[0] += 0.11926283945584981;
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
        result[0] += -0.001156972815214813;
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
              result[0] += 0.02853349860304758;
            } else {
              result[0] += -0.0003587581714063428;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
              result[0] += -0.009365947457802238;
            } else {
              result[0] += -0.05730057683652458;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            result[0] += -0.007970544883247565;
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0005339226717322858;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.497191667556763583) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
                    result[0] += -0.006248571882710871;
                  } else {
                    result[0] += 0.014929341784151787;
                  }
                } else {
                  result[0] += 0.03411645104024161;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.176976680755616123) ) ) {
                    result[0] += -0.04976618732438601;
                  } else {
                    result[0] += -0.0074349969176410865;
                  }
                } else {
                  result[0] += 0.008319949428042444;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.02973808741480052;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)57.50000000000000711) ) ) {
              result[0] += -0.0686327359940646;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.052282205042943064;
              } else {
                result[0] += -0.005587350200073373;
              }
            }
          }
        } else {
          result[0] += -0.04901379094790366;
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.0004993472305822245;
        } else {
          result[0] += -0.022936852047913243;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.60200452804565607) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.493027687072754794) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.003311278114855532;
          } else {
            result[0] += 0.04393448021685539;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.05317087100054749;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.06887725025815393;
            } else {
              result[0] += -0.008329036687215147;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.854362010955811435) ) ) {
            result[0] += 0.001260713627481829;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)211.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
                    result[0] += -0.0065331638878176395;
                  } else {
                    result[0] += -0.07031399626415945;
                  }
                } else {
                  result[0] += -0.06383520109143588;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.05729720375920211;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.920663833618164951) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
                              result[0] += -0.05306602198015527;
                            } else {
                              result[0] += 0.025259475816855198;
                            }
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += 0.1346043443726628;
                            } else {
                              result[0] += -0.09505736738286563;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.51615929603576749) ) ) {
                            result[0] += 0.005488156749591824;
                          } else {
                            result[0] += -0.06425364642881122;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.06439283204541417;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.017021125807442337;
                      } else {
                        result[0] += -0.00832807729724972;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                        if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.011069845380764342;
                        } else {
                          result[0] += -0.058349434072889686;
                        }
                      } else {
                        result[0] += 0.013324768487756808;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                      result[0] += 0.014128128683034681;
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.030284570906573823;
                      } else {
                        result[0] += -0.004176953532262133;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.03216550846425353;
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                            result[0] += 0.045966777244996045;
                          } else {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.997585535049439365) ) ) {
                                result[0] += 0.004447617382498305;
                              } else {
                                result[0] += -0.08365274891161029;
                              }
                            } else {
                              result[0] += 0.03932554064920796;
                            }
                          }
                        } else {
                          result[0] += 0.07324455561335334;
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)141.5000000000000284) ) ) {
                              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)140.5000000000000284) ) ) {
                                result[0] += -0.0023087695759327594;
                              } else {
                                result[0] += 0.0735379470717118;
                              }
                            } else {
                              result[0] += -0.04967849237085309;
                            }
                          } else {
                            result[0] += 0.038091352746365575;
                          }
                        } else {
                          result[0] += 0.05576716677404268;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.04634054689170278;
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.005145657832353793;
          } else {
            result[0] += -0.023669573116975262;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
        result[0] += -0.13890754798846294;
      } else {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
            result[0] += -0.08024923863408175;
          } else {
            result[0] += -0.0034688416863745877;
          }
        } else {
          result[0] += -0.005764065746110287;
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
        result[0] += -0.0011304093973405899;
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.609406948089601386) ) ) {
            result[0] += 0.002325583962678923;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.01235967196019319;
              } else {
                result[0] += -0.049359857084419034;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95053911209106623) ) ) {
                result[0] += 0.004222231948953126;
              } else {
                result[0] += 0.0686372612597148;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.549646615982056552) ) ) {
            result[0] += -0.009886743794827669;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0035088140003991376;
              } else {
                result[0] += 0.015508835571570116;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.548691272735597479) ) ) {
                result[0] += -0.03230489934444784;
              } else {
                result[0] += 0.006393663143346708;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.009649265300869033;
        } else {
          result[0] += -0.028521840819976038;
        }
      } else {
        result[0] += -0.0011375856440250375;
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
        result[0] += -0.081888127526401;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.0019705376698743683;
          } else {
            result[0] += 0.028093512365633767;
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
                      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.132412433624269354) ) ) {
                          result[0] += -0.23032897434935143;
                        } else {
                          result[0] += 0.0183808152273613;
                        }
                      } else {
                        result[0] += -0.001973730150024199;
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.960767745971680132) ) ) {
                        result[0] += 0.013760920351154003;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          result[0] += 0.09665644571312249;
                        } else {
                          result[0] += -0.02231077266926072;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
                            result[0] += 0.12942701470460527;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.05705032725505832;
                            } else {
                              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += 0.11952817389144338;
                              } else {
                                result[0] += 0.00959986573868386;
                              }
                            }
                          }
                        } else {
                          result[0] += -0.062322086769756015;
                        }
                      } else {
                        result[0] += -0.05560051904309459;
                      }
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.00579018098241783;
                        } else {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                            result[0] += -0.04103026466425123;
                          } else {
                            result[0] += -0.01011257207067502;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)37.50000000000000711) ) ) {
                            result[0] += -0.00154043003338274;
                          } else {
                            result[0] += 0.016654233343493918;
                          }
                        } else {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.0163196399262353;
                          } else {
                            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                              result[0] += 0.0010194607082609451;
                            } else {
                              result[0] += -0.03212182693558471;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.048147007245494394;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.0250784163356537;
                    } else {
                      result[0] += 0.005453322785593756;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)9.322573661804200995) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.18732333183288663) ) ) {
                      result[0] += -0.0019922440503531876;
                    } else {
                      result[0] += -0.0748163944354019;
                    }
                  } else {
                    result[0] += 0.648801776690035;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                    result[0] += -0.0234936089139745;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)121.5000000000000142) ) ) {
                      result[0] += 0.09416389663129075;
                    } else {
                      result[0] += -0.019868175342594666;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.07397255614183394;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.448499202728272373) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0437333122018009;
              } else {
                result[0] += -0.035964070944371644;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.1367622084003601;
                    } else {
                      result[0] += -0.049666801936796345;
                    }
                  } else {
                    result[0] += -0.10517674280304493;
                  }
                } else {
                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04231097188672177;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.01364113251412657;
                    } else {
                      result[0] += -0.029202561773207528;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.649175643920899326) ) ) {
                  result[0] += 0.008291598617719078;
                } else {
                  result[0] += -0.04464649267385061;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
            result[0] += 0.010544409508207633;
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08583666210270982;
              } else {
                result[0] += -0.016466695090424616;
              }
            } else {
              result[0] += 0.0588610514302354;
            }
          }
        } else {
          result[0] += 0.03443232208713178;
        }
      } else {
        result[0] += 0.060742152619292604;
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.24492526054382413) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.019022396480156144;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.001684526007904133;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += 0.00198320650016485;
                    } else {
                      result[0] += -0.09599534298460359;
                    }
                  } else {
                    result[0] += 0.022850867374979798;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                  if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.008913318251951045;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
                      result[0] += -0.02530344277193108;
                    } else {
                      result[0] += 0.0833508735312441;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
                        result[0] += -0.1289587402151539;
                      } else {
                        result[0] += -0.00395139508312676;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.597218394279480425) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.137252807617188388) ) ) {
                          result[0] += -0.01564880184259004;
                        } else {
                          result[0] += 0.05333634011350411;
                        }
                      } else {
                        result[0] += 0.041721441064106995;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.0042232961955067025;
                    } else {
                      result[0] += -0.011242276988575042;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)257.5000000000000568) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.551017761230469638) ) ) {
                    result[0] += 0.0010414800365896094;
                  } else {
                    result[0] += 0.019458993625540367;
                  }
                } else {
                  result[0] += -0.04493401137889533;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += -0.0001090219749488883;
                } else {
                  result[0] += 0.07986466669503817;
                }
              } else {
                result[0] += -0.007923827905786508;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.569529533386231357) ) ) {
                    if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.016704786175885455;
                    } else {
                      result[0] += 0.017920404609306852;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += 0.003916047934759876;
                    } else {
                      result[0] += 0.03883671942087167;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                    result[0] += -0.08940009351416173;
                  } else {
                    result[0] += 0.030481120804613977;
                  }
                }
              } else {
                result[0] += -0.022902251989917292;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.010463726466597236;
            } else {
              result[0] += 0.0016075408423323178;
            }
          } else {
            result[0] += -0.037392066303188094;
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.610357046127320224) ) ) {
            result[0] += -0.0008235262415583781;
          } else {
            result[0] += 0.016437327277075503;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += -0.029370015311245452;
            } else {
              result[0] += -0.0013544936326370784;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.014974609479404159;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.012821303340216045;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += 0.07875174526641472;
                } else {
                  result[0] += 0.010411322298697782;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.944137096405030185) ) ) {
          result[0] += -0.013403961059667633;
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.06776656224804602;
          } else {
            result[0] += 0.042294139350059705;
          }
        }
      } else {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
          result[0] += 0.09034132701286791;
        } else {
          result[0] += 0.0031664616245151212;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
      result[0] += 0.00227284522088276;
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.758822202682496005) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.686739683151246005) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)155.5000000000000284) ) ) {
              result[0] += -0.0003707436921206429;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += -0.042191891770685615;
              } else {
                result[0] += -0.007397341882474322;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
              result[0] += -0.031769092338315465;
            } else {
              result[0] += 0.045903892999929696;
            }
          }
        } else {
          result[0] += -0.031012259958352925;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.06526310849635976;
                } else {
                  result[0] += 0.08482389021320574;
                }
              } else {
                result[0] += -0.025821135543873316;
              }
            } else {
              result[0] += -0.10495004471604484;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)225.5000000000000284) ) ) {
              result[0] += 0.0026111911333966077;
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.478551149368287021) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.01553200576472909;
                } else {
                  result[0] += -0.018323487553473083;
                }
              } else {
                result[0] += -0.04445606928790791;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.061067788511153014;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.07462948761709898;
              } else {
                result[0] += -0.09411513115967399;
              }
            }
          } else {
            result[0] += -0.057176877922192164;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)49.50000000000000711) ) ) {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.937313556671143466) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)44.50000000000000711) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.673299551010132724) ) ) {
                            result[0] += -0.014335425554325935;
                          } else {
                            result[0] += -0.0568322283357007;
                          }
                        } else {
                          result[0] += 0.02350190663773732;
                        }
                      } else {
                        result[0] += 0.0022327919185171625;
                      }
                    } else {
                      result[0] += -0.05229589835735975;
                    }
                  } else {
                    result[0] += -0.05488169223717319;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
                      result[0] += -0.04228106816009983;
                    } else {
                      result[0] += 0.04188295905144374;
                    }
                  } else {
                    result[0] += -0.01732858483259017;
                  }
                }
              } else {
                result[0] += 0.002773704045826053;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                  result[0] += -0.015951589098944053;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.020509960492660826;
                  } else {
                    result[0] += -0.06794677365468094;
                  }
                }
              } else {
                result[0] += -0.038895899763641786;
              }
            }
          } else {
            result[0] += 0.002515284383297324;
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)28.50000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
              result[0] += -0.010764677191800678;
            } else {
              result[0] += 0.002896689578754682;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58603620529174982) ) ) {
                result[0] += 0.03091835274349976;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                  result[0] += 0.02915416109121538;
                } else {
                  result[0] += -0.024875878862576556;
                }
              }
            } else {
              result[0] += 0.00414198159470315;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.551017761230469638) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    result[0] += -0.022130044606365994;
                  } else {
                    result[0] += 0.055436150176340765;
                  }
                } else {
                  result[0] += -0.02537902432494557;
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                    result[0] += -0.05782347524621502;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.280697107315064365) ) ) {
                      result[0] += -0.08168088642965188;
                    } else {
                      result[0] += 0.030657717873763438;
                    }
                  }
                } else {
                  result[0] += 0.12723824236741513;
                }
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += -0.02496453725752088;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0554502174113716;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.89399480819702326) ) ) {
                      result[0] += -0.02198820782860243;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
                        result[0] += 0.08788075602816481;
                      } else {
                        result[0] += 0.011473991960700022;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.02020045085163655;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
                  result[0] += 0.050784052009949746;
                } else {
                  result[0] += -0.0002828733366836852;
                }
              } else {
                result[0] += 0.03575409808971405;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)13.50000000000000178) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                  result[0] += -0.09693613876771323;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.020971910782152974;
                  } else {
                    result[0] += 0.05018952303941226;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0606849562263455;
                } else {
                  result[0] += -0.02620056115568243;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
            result[0] += 0.005944141190277512;
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.07625834418822701;
            } else {
              result[0] += -0.013157247532342332;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.000307083129883701) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)81.50000000000001421) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)80.50000000000001421) ) ) {
            result[0] += 0.007286486917945001;
          } else {
            result[0] += 0.0420245517847257;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
            result[0] += -0.01570796427273027;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)95.50000000000001421) ) ) {
              result[0] += 0.022855576079529125;
            } else {
              result[0] += -0.0014332412195690331;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.704609394073488104) ) ) {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.473832368850708896) ) ) {
                result[0] += -0.0037416910844025133;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)9.167253971099855292) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.161602735519410068) ) ) {
                        result[0] += -0.0569092851424719;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.118075370788575107) ) ) {
                          result[0] += -0.06122477395706182;
                        } else {
                          result[0] += 0.04349700089906923;
                        }
                      }
                    } else {
                      result[0] += -0.012782751967480908;
                    }
                  } else {
                    result[0] += 0.09331579814350187;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.588751316070558417) ) ) {
                    result[0] += -0.00609085801988812;
                  } else {
                    result[0] += -0.12764674400365594;
                  }
                }
              }
            } else {
              result[0] += -0.030120979022355877;
            }
          } else {
            result[0] += -0.001135550016208584;
          }
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.029114517813051854;
          } else {
            result[0] += 0.040080241933696456;
          }
        }
      }
    }
  } else {
    result[0] += 0.0006337929882346332;
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
      result[0] += -0.0003662337429839995;
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.728756666183472568) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                      result[0] += -0.03936556793844237;
                    } else {
                      result[0] += 0.006927491196371412;
                    }
                  } else {
                    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.055793118556673287;
                    } else {
                      result[0] += 0.010184066226902835;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    result[0] += -0.05082795422596981;
                  } else {
                    result[0] += 0.015240631953718121;
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.623641014099121982) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.632002353668214667) ) ) {
                    result[0] += -0.0411882057117851;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.260092735290528232) ) ) {
                      result[0] += -0.048437010447631774;
                    } else {
                      result[0] += 0.03697373764913494;
                    }
                  }
                } else {
                  result[0] += 0.06591167830874094;
                }
              }
            } else {
              result[0] += -0.026822003381473697;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.308072090148926669) ) ) {
              result[0] += -0.028348794793478956;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.05549950550769364;
              } else {
                result[0] += 0.02268870388402302;
              }
            }
          }
        } else {
          result[0] += -0.06650617637966925;
        }
      } else {
        result[0] += -0.021102952814383628;
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.2115969657897967) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.847910165786744052) ) ) {
            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.002091661649641368;
            } else {
              result[0] += 0.019131759319709344;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)226.5000000000000284) ) ) {
                result[0] += -0.0012973976714157366;
              } else {
                result[0] += 0.031783430286643606;
              }
            } else {
              result[0] += 0.019130788054076767;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.854362010955811435) ) ) {
              result[0] += 0.0009640921530234408;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)211.5000000000000284) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.24492526054382413) ) ) {
                        result[0] += -0.004625300152627471;
                      } else {
                        result[0] += -0.07478032554220485;
                      }
                    } else {
                      result[0] += -0.05489398590346463;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.005060418387278307;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.728756666183472568) ) ) {
                          result[0] += -0.03219520979067846;
                        } else {
                          result[0] += 0.0473500654358783;
                        }
                      }
                    } else {
                      result[0] += 0.0035684056623578237;
                    }
                  }
                } else {
                  result[0] += -0.04758494875418712;
                }
              } else {
                result[0] += -0.047788744053431774;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)40.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.200417995452881748) ) ) {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.005164104011908191;
                    } else {
                      result[0] += -0.06914114735560777;
                    }
                  } else {
                    result[0] += 0.0017388980663573834;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.099201440811158115) ) ) {
                    result[0] += 0.02062807173992088;
                  } else {
                    result[0] += -0.030296780836109944;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.319199085235596591) ) ) {
                  result[0] += 0.016030729199406957;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                    result[0] += 0.01869962181373304;
                  } else {
                    result[0] += -0.024355370703151663;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.007297440561663472;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.987184524536133701) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.04648788905345316;
                  } else {
                    result[0] += 0.000928479886580501;
                  }
                } else {
                  result[0] += -0.00523553112208583;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                result[0] += -0.011633459879744314;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.0019006107083874666;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                      result[0] += 0.006453389621744546;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                        result[0] += -0.07228850951308555;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.06771350959011123;
                        } else {
                          result[0] += 0.02398232941695067;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.825982809066773349) ) ) {
                      result[0] += -0.007512652143186454;
                    } else {
                      result[0] += -0.04521300678861162;
                    }
                  } else {
                    result[0] += 0.010117287081631066;
                  }
                }
              }
            } else {
              result[0] += -0.028809255696215466;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                result[0] += 0.05522749542253761;
              } else {
                result[0] += -0.06309994148491523;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.03558289215931347;
                } else {
                  result[0] += 0.030106087473034983;
                }
              } else {
                result[0] += -0.08478974627747271;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.029498418357034303;
          } else {
            result[0] += 0.004175778814148885;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 0.00853284465297987;
      } else {
        result[0] += 0.0532688155458613;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
          result[0] += 0.0005878511690504216;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.020049625799479748;
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.11425846883852225;
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                        result[0] += 0.011068732765006107;
                      } else {
                        result[0] += -0.03511051913658018;
                      }
                    }
                  } else {
                    result[0] += -0.06753476803241287;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.763591527938843662) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                      result[0] += -0.009588382804306255;
                    } else {
                      result[0] += -0.03135395471057651;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05711337425415752;
                      } else {
                        result[0] += -0.008214299592697637;
                      }
                    } else {
                      result[0] += 0.007199818357628636;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.8328428268432635) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.590987443923951083) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.075335502624512607) ) ) {
                        result[0] += -0.044905005814233845;
                      } else {
                        result[0] += 0.004265702726062083;
                      }
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.007903065478529068;
                      } else {
                        result[0] += 0.04558896197337259;
                      }
                    }
                  } else {
                    result[0] += 0.05865303048177759;
                  }
                } else {
                  result[0] += -0.009258927168526786;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.03895473480224787) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.455312013626099521) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.908255100250245029) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.015528270128082429;
                    } else {
                      result[0] += -0.05443414499244077;
                    }
                  } else {
                    result[0] += 0.0461507396368879;
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.169734954833985263) ) ) {
                    if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.0776431013148214;
                    } else {
                      result[0] += 0.018938956377697783;
                    }
                  } else {
                    result[0] += -0.01169835200807738;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                    result[0] += 0.013790037028884745;
                  } else {
                    result[0] += -0.03813594297861534;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.221583127975464755) ) ) {
                      result[0] += -0.008849989850399613;
                    } else {
                      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.03955724592872703;
                      } else {
                        result[0] += -0.10393756045855003;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.193383932113648349) ) ) {
                        result[0] += 0.01999378393178273;
                      } else {
                        result[0] += -0.04102885666609614;
                      }
                    } else {
                      result[0] += 0.017444351778808317;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.07734432088058144;
              } else {
                result[0] += -0.013359110014242335;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.182021141052246982) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.941534638404846635) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                      result[0] += -0.017426714611649543;
                    } else {
                      result[0] += -0.060242200566814785;
                    }
                  } else {
                    if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                        result[0] += 0.05802898970573275;
                      } else {
                        result[0] += 0.006306525354204159;
                      }
                    } else {
                      result[0] += -0.0056980083747657515;
                    }
                  }
                } else {
                  result[0] += 0.005159175231013186;
                }
              } else {
                result[0] += 0.008327227009197307;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.973273515701294833) ) ) {
                result[0] += -0.0029453718629527457;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.007507836058467211;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.06610402689584045;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                      result[0] += -0.013065503706003177;
                    } else {
                      result[0] += -0.09696982380054059;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0024871474256397794;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.02729072041713212;
                  } else {
                    result[0] += 0.05938837034248923;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                    result[0] += 0.08088309749143163;
                  } else {
                    result[0] += -0.005024607622650348;
                  }
                } else {
                  result[0] += -0.05467771707097207;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.0003579893993087152;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
                  result[0] += -0.005091913372772807;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                      result[0] += 0.01298112060729919;
                    } else {
                      result[0] += -0.07964011920621318;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.05856250830548873;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                        if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.037901903843447604;
                        } else {
                          result[0] += -0.03457296092881084;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                          result[0] += 0.10840770337414361;
                        } else {
                          result[0] += -0.04472782352934094;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.002370189223411869;
        }
      }
    } else {
      result[0] += -0.017814985106213804;
    }
  } else {
    result[0] += 0.000747592866084352;
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.309873342514038974) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -5.6063400890863835e-05;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.344720840454102451) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
              result[0] += -0.0013053915401293424;
            } else {
              result[0] += -0.044120266151619285;
            }
          } else {
            result[0] += -0.030446351345682993;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
            result[0] += -0.0029831940051233707;
          } else {
            result[0] += 0.041657511751793774;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
        result[0] += -0.01715036331835295;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.01041157270075811;
        } else {
          result[0] += 0.006501380960831112;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.007252185700530981;
              } else {
                result[0] += -0.0236331653616747;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.009368638090219933;
                    } else {
                      result[0] += -0.06199484739380329;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.055890219417530085;
                    } else {
                      result[0] += 0.023509306307171445;
                    }
                  }
                } else {
                  result[0] += 0.0032124879854380922;
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08236300274985194;
                  } else {
                    result[0] += -0.02330114375967088;
                  }
                } else {
                  result[0] += 0.05519321058817901;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.831219434738160068) ) ) {
              result[0] += -0.021055472688578103;
            } else {
              result[0] += 0.036018903342851204;
            }
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.126931190490723544) ) ) {
                    result[0] += 0.0013433617459327693;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
                      result[0] += -0.001171528783823366;
                    } else {
                      result[0] += -0.00797581445616522;
                    }
                  }
                } else {
                  result[0] += 0.0073818594475988115;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)16.50000000000000355) ) ) {
                  result[0] += -0.018660441591832135;
                } else {
                  result[0] += 0.006624877845680409;
                }
              }
            } else {
              result[0] += 0.007707587534514764;
            }
          } else {
            result[0] += 0.016359791365405295;
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
              result[0] += 0.015459444603350825;
            } else {
              result[0] += -0.07522708973500553;
            }
          } else {
            result[0] += 0.039576695781664;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.559112548828125888) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                      result[0] += -0.0007856787898097971;
                    } else {
                      result[0] += -0.04630477470502489;
                    }
                  } else {
                    if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.025410435387588567;
                    } else {
                      result[0] += -0.011110562594927341;
                    }
                  }
                } else {
                  result[0] += -0.032867312900179274;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += -0.01854557974351353;
                } else {
                  result[0] += -0.05353313325494934;
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                    result[0] += -0.008062703405986447;
                  } else {
                    result[0] += 0.008069802189354621;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.02705633837345749;
                    } else {
                      result[0] += -0.005103212118840999;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.342765808105469638) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.018998295404365914;
                          } else {
                            result[0] += -0.0739345679511061;
                          }
                        } else {
                          result[0] += -0.0566171799777887;
                        }
                      } else {
                        result[0] += -0.021872599324773062;
                      }
                    } else {
                      result[0] += 0.029370512447757485;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.1589025979917291;
                    } else {
                      result[0] += -0.09425823895975538;
                    }
                  } else {
                    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.054283901190938504;
                    } else {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.04282140016702494;
                      } else {
                        result[0] += 0.0003764628380367475;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.037478617977367616;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
                      result[0] += 0.005556856098703816;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.019340026564949035;
                            } else {
                              result[0] += 0.1267992207313053;
                            }
                          } else {
                            result[0] += 0.06153620113288155;
                          }
                        } else {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.03045587487889801;
                          } else {
                            result[0] += 0.06013513020364574;
                          }
                        }
                      } else {
                        result[0] += 0.10397696192727868;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.03862960574034738;
          }
        }
      }
    } else {
      result[0] += 0.011201214083197375;
    }
  }
  if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.309873342514038974) ) ) {
      result[0] += -0.0002912733311572345;
    } else {
      result[0] += -0.008705543887197888;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)235.5000000000000284) ) ) {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.342765808105469638) ) ) {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.026870651397225054;
                  } else {
                    result[0] += -0.04652526026915192;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += 0.008015499792260756;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                          result[0] += -0.0331858282394747;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                            result[0] += -0.0029208074657795105;
                          } else {
                            result[0] += -0.018148680506000447;
                          }
                        }
                      } else {
                        result[0] += 0.01185982734513484;
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                          result[0] += 0.0029196418866231813;
                        } else {
                          result[0] += 0.02026789615849662;
                        }
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.480810642242432529) ) ) {
                          result[0] += -0.01651564059549544;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                              result[0] += 0.00020244747545928324;
                            } else {
                              result[0] += -0.07741985440470177;
                            }
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                              result[0] += 0.0038551780374586468;
                            } else {
                              result[0] += 0.03210242388185159;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
                    result[0] += 0.009586216972263622;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.0026389236820550683;
                    } else {
                      result[0] += -0.07825398845286821;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                    result[0] += 0.027732151270247363;
                  } else {
                    result[0] += -0.06862543830932348;
                  }
                }
              }
            } else {
              result[0] += 0.023183153242424315;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                result[0] += -0.031848052809013434;
              } else {
                result[0] += 0.002446221410311095;
              }
            } else {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.09211438005618906;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                  result[0] += -0.07894421961548893;
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.03491501828178197;
                  } else {
                    result[0] += -0.05565061287554214;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += 0.044416955179668455;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                result[0] += 0.053122968830346964;
              } else {
                result[0] += -0.03105396921793779;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                result[0] += -0.03175357901050234;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                  result[0] += 0.027278381610021635;
                } else {
                  result[0] += -0.011977994580217505;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.01372264756234126;
      }
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
          result[0] += 0.004904327236847998;
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
              result[0] += -0.0034551218101213075;
            } else {
              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.603942871093750888) ) ) {
                      result[0] += 0.0026983292879702607;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.137252807617188388) ) ) {
                        result[0] += -0.02606852544349336;
                      } else {
                        result[0] += -0.08695276534747151;
                      }
                    }
                  } else {
                    result[0] += -0.11935451261794697;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                    result[0] += -0.05823963519777608;
                  } else {
                    result[0] += -0.1225630336420744;
                  }
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338887453079224521) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.854362010955811435) ) ) {
                      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.308072090148926669) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                            result[0] += -0.021748046282342157;
                          } else {
                            result[0] += 0.04875116483307256;
                          }
                        } else {
                          result[0] += 0.04891664373273863;
                        }
                      } else {
                        result[0] += -0.013148773533255553;
                      }
                    } else {
                      result[0] += -0.024561523506551865;
                    }
                  } else {
                    result[0] += -0.053857941588691886;
                  }
                } else {
                  result[0] += -0.06595644884856317;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.548691272735597479) ) ) {
              result[0] += 0.014956372368004381;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                result[0] += -0.0005469738605553584;
              } else {
                result[0] += -0.025506484577135436;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.006028328364569124;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
              result[0] += -0.09882036308628034;
            } else {
              result[0] += 0.06967474890827445;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
              result[0] += 0.0047154513275742115;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
                result[0] += -0.03588241860269415;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)149.5000000000000284) ) ) {
                  result[0] += 0.0017082154838436916;
                } else {
                  result[0] += -0.051577667422792406;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
              result[0] += -0.005560762885707689;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.034729620798999085;
              } else {
                result[0] += 0.017689689742864983;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.309873342514038974) ) ) {
      result[0] += -0.0003470179304221942;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
        result[0] += -0.016429032876392965;
      } else {
        result[0] += -0.00284455290840513;
      }
    }
  } else {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
        result[0] += -0.07423398152474227;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11214685440063654) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                              result[0] += 0.01311981549581774;
                            } else {
                              result[0] += -0.110601198409041;
                            }
                          } else {
                            result[0] += 0.00012937432759942054;
                          }
                        } else {
                          result[0] += -0.0006384368287752316;
                        }
                      } else {
                        if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.005821263146845597;
                        } else {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)67.50000000000001421) ) ) {
                            result[0] += 0.0457650047061884;
                          } else {
                            result[0] += -0.019889271852678844;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.013147818274541201;
                    }
                  } else {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.04206759735785749;
                    } else {
                      result[0] += -0.00357319671603332;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                    result[0] += -0.001655332818981289;
                  } else {
                    result[0] += 0.015412400408324117;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
                    result[0] += 0.005256091874102139;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.04324212894291096;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                            result[0] += 0.05474620820297582;
                          } else {
                            result[0] += 0.14723843805860526;
                          }
                        }
                      } else {
                        result[0] += 0.01934551040371472;
                      }
                    } else {
                      result[0] += 0.01348266517082631;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.31479072570800959) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                            result[0] += -0.050697597747156466;
                          } else {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.009267710680305805;
                            } else {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.389061450958252841) ) ) {
                                result[0] += -0.067694404528898;
                              } else {
                                result[0] += 0.007391570340184519;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.0142437665526315;
                        }
                      } else {
                        result[0] += 0.011654168720070454;
                      }
                    } else {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)256.5000000000000568) ) ) {
                          result[0] += 0.030931343959787652;
                        } else {
                          result[0] += 0.1319002987155495;
                        }
                      } else {
                        result[0] += 0.007396189588237992;
                      }
                    }
                  } else {
                    result[0] += -0.019217226887914992;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.002494412524734999;
              } else {
                if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.004777009510367039;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.03002367791888413;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
                      result[0] += -0.022367855514825744;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                          result[0] += -0.07307190283309375;
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.10052819767041339;
                          } else {
                            result[0] += -0.008440285570879696;
                          }
                        }
                      } else {
                        result[0] += -0.002357321147118951;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.031550313398577105;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.758822202682496005) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.132848501205445224) ) ) {
                    result[0] += -0.015285977676017718;
                  } else {
                    result[0] += 0.03830931818269696;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.132848501205445224) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.043138764538776825;
                    } else {
                      result[0] += -0.012981405883437814;
                    }
                  } else {
                    result[0] += -0.010521752860244414;
                  }
                }
              } else {
                result[0] += -0.02674440297491376;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.06921265254240447;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                      result[0] += -0.01771538866830425;
                    } else {
                      result[0] += -0.05622616126830352;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.010878397981977223;
                  } else {
                    result[0] += -0.03740422665918891;
                  }
                }
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.019055874814354685;
                } else {
                  result[0] += -0.04871668829207618;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.276966691017151323) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.05577433756167946;
                  } else {
                    result[0] += 0.08752208072752567;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += -0.021740109742587405;
                  } else {
                    result[0] += -0.08396368673272758;
                  }
                }
              } else {
                result[0] += 0.001580213780578713;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04568371231429999;
                } else {
                  result[0] += 0.033732691667633874;
                }
              } else {
                result[0] += -0.0719993470690508;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.010772243238845023;
    }
  }
}

