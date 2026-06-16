
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
        result[0] += 0.009554286175218626;
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.597218394279480425) ) ) {
          result[0] += -0.002747625187498309;
        } else {
          result[0] += -0.01654036019883871;
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)89.50000000000001421) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)88.50000000000001421) ) ) {
          result[0] += 0.0005460100738697671;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
            result[0] += -0.02470887272561111;
          } else {
            result[0] += 0.036507518491902945;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.000802633321365051;
          } else {
            result[0] += -0.017328375811727047;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += -0.00397100400510249;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.06291471743312181;
              } else {
                result[0] += 0.04488842409384027;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0439222840653944;
              } else {
                result[0] += -0.008795032808486201;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)238.5000000000000284) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.666320323944092685) ) ) {
                result[0] += -0.007197052349389387;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.006608150020844934;
                } else {
                  result[0] += 0.03960357984493905;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.09486948783641075;
              } else {
                result[0] += -0.018815624543899928;
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.03733245299654843;
            } else {
              result[0] += 0.006362789055546162;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
            result[0] += 0.008567709941751697;
          } else {
            result[0] += 0.026989214092356495;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)246.5000000000000284) ) ) {
              result[0] += -0.0005260017133021514;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                result[0] += -0.020469931748640684;
              } else {
                result[0] += -0.0020103283370692076;
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.591613531112671787) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.026372191909253957;
                  } else {
                    result[0] += -0.014694793301716212;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.03207424376675328;
                  } else {
                    result[0] += -0.027904312284536877;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.06464824897193183;
                } else {
                  result[0] += -0.02769568956879442;
                }
              }
            } else {
              result[0] += -0.047414290923828886;
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)195.5000000000000284) ) ) {
            result[0] += 0.02259713428058102;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.869741916656495029) ) ) {
              result[0] += 0.012684070662765348;
            } else {
              result[0] += -0.017598120960319093;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
          result[0] += -0.007310355176437274;
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8192477226257342) ) ) {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += 0.003354331915471929;
                      } else {
                        result[0] += -0.062088252902707565;
                      }
                    } else {
                      result[0] += 0.05784308109333162;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.040111042493625924;
                    } else {
                      result[0] += 0.0035861382685599985;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.006269758754138665;
                  } else {
                    result[0] += 0.04775453693551959;
                  }
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.000429477949166564;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.923617362976075107) ) ) {
                      result[0] += 0.029253933864159743;
                    } else {
                      result[0] += 0.06370931584408406;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.03473745799850352;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.0461232320736884;
                    } else {
                      result[0] += 0.0416675130417144;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.05465651356067966;
                  } else {
                    result[0] += 0.1210063434174231;
                  }
                } else {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                      result[0] += -0.05684995107216045;
                    } else {
                      result[0] += 0.09787266529551579;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                      result[0] += 0.046974961710577204;
                    } else {
                      result[0] += -0.012687025970448166;
                    }
                  }
                }
              } else {
                result[0] += 0.010004593572858815;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.0065147324823093215;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                result[0] += -0.010300851585752481;
              } else {
                result[0] += 0.07493834593349466;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += -0.06718283355327576;
          } else {
            result[0] += 0.09186169604377242;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)130.5000000000000284) ) ) {
            result[0] += -0.07562745456754064;
          } else {
            result[0] += -0.01227065625032498;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.11285294532647738;
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.422742605209351474) ) ) {
                          result[0] += -0.1383755004882896;
                        } else {
                          result[0] += 0.06233746012755949;
                        }
                      } else {
                        result[0] += 0.06716294737752;
                      }
                    } else {
                      result[0] += 0.009976545915398139;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.283562898635865146) ) ) {
                        result[0] += 0.005514619413420678;
                      } else {
                        result[0] += -0.05726991430283134;
                      }
                    } else {
                      result[0] += 0.016916336674828975;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.018095665322219078;
                  } else {
                    result[0] += 0.008952589211318333;
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.350240230560303178) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += -0.020149665200635256;
                  } else {
                    result[0] += -0.0012500043132643438;
                  }
                } else {
                  result[0] += 0.039923118291214005;
                }
              }
            } else {
              result[0] += 0.02531894651686771;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.132848501205445224) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.014575308753621173;
                } else {
                  result[0] += -0.07006003243312914;
                }
              } else {
                result[0] += 0.00023848672758030278;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.868494272232056552) ) ) {
                    result[0] += -0.07654810587156695;
                  } else {
                    result[0] += -0.01249937379178756;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.01902349534890864;
                    } else {
                      result[0] += -0.015833416820830247;
                    }
                  } else {
                    result[0] += -0.06117982080952768;
                  }
                }
              } else {
                result[0] += -0.02899810485228707;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.714014530181885654) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += 0.06798607651054481;
                } else {
                  result[0] += -0.010044007646594036;
                }
              } else {
                result[0] += -0.037250047588044864;
              }
            } else {
              result[0] += -0.06604619458867086;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.406318187713624823) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.505334615707398349) ) ) {
                result[0] += -0.005375777185599884;
              } else {
                result[0] += 0.011049246736125227;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.048558144146253424;
                  } else {
                    result[0] += -0.02293353102350749;
                  }
                } else {
                  result[0] += 0.09045232942194387;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.01560926102360869;
                } else {
                  result[0] += 0.02846011542777533;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.166635274887085849) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.026573108584381746;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.075206041336061347) ) ) {
              result[0] += -0.0007284862837331064;
            } else {
              result[0] += 0.05797159518851173;
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += 0.015887486992237257;
              } else {
                result[0] += 0.07174305612635011;
              }
            } else {
              result[0] += -0.061188433946673886;
            }
          } else {
            result[0] += 0.008160282312462073;
          }
        }
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0043232956170474795;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.009634721938542269;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += -0.013360099709973698;
              } else {
                result[0] += 0.0632225515020751;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += 0.012168150468589723;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.09310344671137707;
                    } else {
                      result[0] += -0.02100550095367566;
                    }
                  } else {
                    result[0] += 0.027374501161582545;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.06947415973831549;
                  } else {
                    result[0] += 0.01436968858473055;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.031105573553591337;
                  } else {
                    result[0] += 0.022996112282792018;
                  }
                }
              }
            } else {
              result[0] += -0.009416322686127432;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.012249944726871078;
              } else {
                result[0] += -0.0634987980014203;
              }
            } else {
              result[0] += 0.03305268141430399;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.005566893127432705;
      } else {
        result[0] += -0.04714360238063123;
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
        result[0] += 0.000733676873673597;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.005660495047133049;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
            result[0] += -0.012728844504654724;
          } else {
            result[0] += -0.05299920982749713;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)77.50000000000001421) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += -0.013413292233487277;
              } else {
                result[0] += -0.04957768525097773;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.280659198760987216) ) ) {
                result[0] += -0.02626533147269818;
              } else {
                result[0] += 0.0267819267424676;
              }
            }
          } else {
            result[0] += -0.004465595362824386;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                result[0] += -0.0009478984519981327;
              } else {
                result[0] += -0.02709990214432304;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.06498757696074962;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.02482396051682355;
                } else {
                  result[0] += -0.08377329304707465;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.006141962379238466;
                } else {
                  result[0] += 0.03723318639556721;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.04372673127661788;
                } else {
                  if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.022134863690166963;
                  } else {
                    result[0] += -0.0042910196353804705;
                  }
                }
              }
            } else {
              result[0] += -0.009266645947835633;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.03255030095684734;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.009802362656545318;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += 0.010606937836886005;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.501469135284425604) ) ) {
                    result[0] += 0.0020460039822370133;
                  } else {
                    result[0] += -0.06641134954715898;
                  }
                }
              }
            } else {
              result[0] += -0.046272948731747233;
            }
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0038299481140316033;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.018671577230674603;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                result[0] += -0.004564233008076406;
              } else {
                result[0] += 0.03464336205160598;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
        result[0] += 0.01246259166323808;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.01302061195376158;
            } else {
              result[0] += -0.0920152155212535;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.06971789784352865;
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.002839204785769082;
              } else {
                result[0] += -0.07598111425230739;
              }
            }
          }
        } else {
          result[0] += 0.03935466152691062;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.016975190341854445;
            } else {
              result[0] += 0.005090928111550141;
            }
          } else {
            result[0] += -0.0455630678585834;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
            result[0] += -0.010372745236808019;
          } else {
            result[0] += -0.04986606721172145;
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
            result[0] += 0.01694355928728201;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.280697107315064365) ) ) {
              result[0] += 0.008984484532075772;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                result[0] += -0.034095945828689876;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  result[0] += 0.026107099625549347;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.0025300084625116816;
                  } else {
                    result[0] += -0.057424543119081475;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
            result[0] += 0.013131934785732782;
          } else {
            result[0] += -0.046075884086330315;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
              result[0] += 0.004790154889499858;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.06643230177951218;
              } else {
                result[0] += -0.010166836589920437;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05471075375801461;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                    result[0] += -0.011880815393508161;
                  } else {
                    result[0] += -0.048710474430117084;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                    result[0] += 0.03997735541758847;
                  } else {
                    result[0] += -0.034855478259838014;
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.0703799232757683;
                  } else {
                    result[0] += -0.006154833318140941;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.004742880622821863;
              } else {
                result[0] += -0.021980533196728456;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
            result[0] += -0.004386531215298371;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += -0.008858431182813998;
            } else {
              result[0] += -0.0347019487278991;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
          result[0] += -0.014134648970179083;
        } else {
          result[0] += -0.04827199116216519;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.02512061248950295;
          } else {
            result[0] += 0.03297121005964909;
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)159.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.0014126204718362843;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += -0.007616043487046094;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0748152633604731;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.113908529281617099) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.05359425699142731;
                    } else {
                      result[0] += -0.01712628226735157;
                    }
                  } else {
                    result[0] += -0.001749538173453796;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.003806691794002515;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.021262253215026953;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  result[0] += -0.014338358846450372;
                } else {
                  result[0] += 0.015548972075924523;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.028226331571343355;
          } else {
            result[0] += -0.0014470979367861175;
          }
        } else {
          result[0] += 0.0028393391914425525;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.022606557305993147;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.07393162966259229;
                } else {
                  result[0] += -0.012163302525920067;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)219.5000000000000284) ) ) {
                  result[0] += -0.002101160556712452;
                } else {
                  result[0] += -0.04355943853232386;
                }
              }
            } else {
              result[0] += 0.05216657683945546;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
            result[0] += 0.036256191761519056;
          } else {
            result[0] += -0.07361337160032204;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03219560046014692;
            } else {
              result[0] += -0.0687782774787728;
            }
          } else {
            result[0] += -0.0017759258191525737;
          }
        } else {
          result[0] += 0.03540812425003761;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.002630173995916552;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
            result[0] += -0.02065486692780007;
          } else {
            result[0] += -0.06952745374808189;
          }
        } else {
          result[0] += 0.0011619552491208436;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
              result[0] += 0.0039837918741061785;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.060477871825971176;
              } else {
                result[0] += 0.011873902194310794;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.05595272671503731;
                    } else {
                      result[0] += -0.06799905369789545;
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.06016028715344133;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40819787979126154) ) ) {
                          result[0] += -0.05626725864090705;
                        } else {
                          result[0] += 0.04295083571493097;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                        result[0] += -0.07372305026217071;
                      } else {
                        result[0] += -0.013438988985262905;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                    result[0] += 0.028923819363788468;
                  } else {
                    result[0] += -0.019954793707864963;
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.04655418209412737;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += 0.05958196356824883;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                      result[0] += -0.07870600858369604;
                    } else {
                      result[0] += -0.008747414317702413;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.767036437988283026) ) ) {
                  result[0] += -0.01860871286925305;
                } else {
                  result[0] += 0.009686825731571924;
                }
              } else {
                result[0] += -0.021088877662323303;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.767036437988283026) ) ) {
                result[0] += 0.025040496099263204;
              } else {
                result[0] += -0.0072343233692936494;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.016637605332352184;
                } else {
                  result[0] += -0.0480085368589074;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.009157358503458385;
                  } else {
                    result[0] += -0.03293928049115535;
                  }
                } else {
                  result[0] += 0.022550004854947306;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03277746839221273;
                } else {
                  result[0] += -0.009744691589242373;
                }
              } else {
                result[0] += -0.03587143827021925;
              }
            } else {
              result[0] += -0.03590783521500475;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.07618183902549808;
        } else {
          result[0] += -0.025876313487535853;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
              result[0] += -0.044029339241443824;
            } else {
              result[0] += 0.02264439094496882;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.59750986099243342) ) ) {
              result[0] += -0.023711801785310165;
            } else {
              result[0] += -0.07270773609356873;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            result[0] += -0.07685045117779224;
          } else {
            result[0] += -0.021992665180063622;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
          result[0] += 0.026108576252654815;
        } else {
          result[0] += -0.008644614347689745;
        }
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.11010603043118045;
          } else {
            result[0] += 0.004269125810331609;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.367881059646607333) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.055629031344300986;
            } else {
              result[0] += 0.02355703207688979;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
              result[0] += 0.1282972503088518;
            } else {
              result[0] += -0.04692467089520935;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
            result[0] += -0.0647912231937759;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.422742605209351474) ) ) {
              result[0] += 0.009992145945498288;
            } else {
              result[0] += -0.059259406762567246;
            }
          }
        } else {
          result[0] += -0.008341206975726509;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)326.5000000000000568) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)81.50000000000001421) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0005121392024241686;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18189048767090021) ) ) {
                    result[0] += -0.024109996040223666;
                  } else {
                    result[0] += 0.015883822568964093;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.07371422627717161;
                          } else {
                            result[0] += -0.009459744289669529;
                          }
                        } else {
                          result[0] += -0.05313452580034798;
                        }
                      } else {
                        result[0] += 0.012616071797375318;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += 0.012264440562125669;
                        } else {
                          result[0] += 0.057174902037264067;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                          result[0] += -0.034250032842198554;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                            result[0] += 0.005907848838883402;
                          } else {
                            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                                result[0] += 0.04619237183328473;
                              } else {
                                result[0] += -0.03703941831129673;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
                                result[0] += 0.008932753009384101;
                              } else {
                                result[0] += 0.066353573355727;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.016900385319948304;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += -0.01594936155064819;
                      } else {
                        result[0] += 0.01809164422324665;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0007008494799506581;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += 0.010419363022288276;
                    } else {
                      result[0] += 0.05793853358592541;
                    }
                  } else {
                    result[0] += -0.0061175726460609495;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.012342691309850847;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
                      result[0] += -0.012099521300323839;
                    } else {
                      result[0] += 0.024066003756089284;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
              result[0] += 0.002575825158960617;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.03653759427122893;
                } else {
                  result[0] += 0.009074815695522277;
                }
              } else {
                result[0] += 0.09105587731135671;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += -0.022396295133498015;
            } else {
              result[0] += 0.04794569168625069;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
              result[0] += -0.023299751394009333;
            } else {
              result[0] += -0.07332720892480228;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += 0.0982069756354158;
        } else {
          result[0] += -0.0163930438760932;
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.39909601211548029) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.285887241363526279) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)141.5000000000000284) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              result[0] += -0.017503762303554268;
            } else {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.1328184115731295;
              } else {
                result[0] += 0.012436627474781607;
              }
            }
          } else {
            result[0] += 0.007435785634799171;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
            result[0] += 0.006069892123719282;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.018685668899622296;
            } else {
              result[0] += -0.066283369726317;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.05534500902725945;
          } else {
            result[0] += 0.00389423835060932;
          }
        } else {
          result[0] += -0.07125204677625606;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.388237953186036044) ) ) {
      result[0] += -0.001505995142280708;
    } else {
      result[0] += -0.008743402783361187;
    }
  } else {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0925779342651385) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.02640807069261941;
            } else {
              result[0] += 0.017399301648725003;
            }
          } else {
            result[0] += 0.03425036916881703;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.014346627769621518;
                } else {
                  result[0] += 0.006920415095384285;
                }
              } else {
                result[0] += 0.016799568016735548;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.0004322360848413169;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.206374883651734287) ) ) {
                  result[0] += -0.008594590263957326;
                } else {
                  result[0] += -0.04817988687932389;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                  result[0] += 0.006212071608206293;
                } else {
                  result[0] += -0.036563426472125106;
                }
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
                    result[0] += -0.00920467684819713;
                  } else {
                    result[0] += -0.06592310077367258;
                  }
                } else {
                  result[0] += -0.005389229034822101;
                }
              }
            } else {
              result[0] += -0.03168760274614564;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += -0.03041008774516877;
                  } else {
                    result[0] += 0.0013042382699381884;
                  }
                } else {
                  result[0] += -0.07556818012000699;
                }
              } else {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    result[0] += -0.013555927240875221;
                  } else {
                    result[0] += 0.007895748677521743;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.0058952392457391764;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += 0.016663497509906724;
                    } else {
                      result[0] += -0.042650226415277;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                result[0] += -0.02611357577045257;
              } else {
                result[0] += 0.025092281622209906;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.013838258189999883;
            } else {
              result[0] += -7.64346003558002e-05;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.012492971734244745;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.009919290850298468;
                } else {
                  result[0] += 0.0007308985439837853;
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.005752249230196252;
                } else {
                  result[0] += 0.017154907339357164;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.03452221739758478;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.0502265240957823;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
                        result[0] += 0.045302977402808815;
                      } else {
                        result[0] += -0.004532704333280198;
                      }
                    }
                  } else {
                    result[0] += -0.04324187573496882;
                  }
                }
              } else {
                result[0] += 0.030626959851674906;
              }
            } else {
              result[0] += 0.046212919256317;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.591613531112671787) ) ) {
        if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.071567356586456743) ) ) {
            result[0] += -0.12133619756329578;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                result[0] += 0.0073038545244238215;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
                  result[0] += -0.11803208279416147;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                        result[0] += -0.009712805616369455;
                      } else {
                        result[0] += -0.06750232308865217;
                      }
                    } else {
                      result[0] += -0.06689127224313936;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                      result[0] += -0.057039821872018885;
                    } else {
                      result[0] += -0.003842864536002995;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0036541191768860873;
                } else {
                  result[0] += -0.042370968993861335;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.005911271016646409;
                  } else {
                    result[0] += -0.028762108553405764;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)180.5000000000000284) ) ) {
                    result[0] += 0.04865991141640279;
                  } else {
                    result[0] += 0.006353117232925193;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.04442545253667296;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)292.5000000000000568) ) ) {
              result[0] += -0.02311953842725205;
            } else {
              result[0] += 0.033334645135659674;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
                result[0] += 0.019744450618302184;
              } else {
                result[0] += -0.07990354686763146;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                result[0] += -0.0753162102488699;
              } else {
                result[0] += 0.05008886762860503;
              }
            }
          }
        } else {
          result[0] += -0.0013664712237844352;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.388237953186036044) ) ) {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
        result[0] += -0.001323923184034183;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.06334612221131364;
        } else {
          result[0] += -0.006359308998096317;
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += 0.014970334071706077;
            } else {
              result[0] += -0.05017884580217552;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
              result[0] += -0.053655590266551034;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += 0.0982221066306305;
              } else {
                result[0] += -0.010296920649778909;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)242.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.914472818374634233) ) ) {
                  result[0] += 0.010070175893506084;
                } else {
                  result[0] += 0.043021516627138295;
                }
              } else {
                result[0] += -0.019556011866366286;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)282.5000000000000568) ) ) {
                result[0] += 0.07855313482432849;
              } else {
                result[0] += 0.009177289327807935;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.11137662044607209;
                } else {
                  result[0] += 0.011828483737624466;
                }
              } else {
                result[0] += 0.02462985413271043;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.431533813476563388) ) ) {
                    result[0] += 0.05911254160672193;
                  } else {
                    result[0] += -0.05989653487936615;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.89342975616455433) ) ) {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.03976256005952084;
                      } else {
                        result[0] += -0.0665181215919769;
                      }
                    } else {
                      result[0] += -0.07122464920275486;
                    }
                  } else {
                    result[0] += 0.07399779573592706;
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.173316955566407138) ) ) {
                      result[0] += -0.0303130909554808;
                    } else {
                      result[0] += -0.10179333179123405;
                    }
                  } else {
                    result[0] += 0.11553262292321223;
                  }
                } else {
                  result[0] += -0.004563047015544718;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
          result[0] += -0.010404443871120545;
        } else {
          result[0] += -0.04089576865772574;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
          result[0] += -0.0165419516595645;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.05557935752318671;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.004751062092479694;
            } else {
              result[0] += 0.03250874046952698;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)326.5000000000000568) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81531858444214045) ) ) {
                result[0] += -0.008544403346951534;
              } else {
                result[0] += 0.030412520173933633;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.36324071884155451) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
                    result[0] += 0.007734522640063681;
                  } else {
                    result[0] += -0.014658850306654767;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                      result[0] += 0.0010839650929605595;
                    } else {
                      result[0] += -0.027030630366312924;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.0014525942452303057;
                    } else {
                      result[0] += -0.03706165762914048;
                    }
                  }
                }
              } else {
                result[0] += -0.018833714010792816;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += 0.00737240731435907;
            } else {
              result[0] += 0.000923665087649894;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += 0.09552567565884909;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.019412609829937217;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                  result[0] += 0.06417804695813682;
                } else {
                  result[0] += -0.009934818322184603;
                }
              }
            } else {
              result[0] += -0.040201805873854336;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += 0.006959009362999091;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.285887241363526279) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += -0.013386361052107706;
              } else {
                result[0] += 0.006974058572105798;
              }
            } else {
              result[0] += 0.030785857626405816;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                  result[0] += -0.0029085914050271727;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.028088950888884807;
                    } else {
                      result[0] += -0.07792237974647115;
                    }
                  } else {
                    result[0] += -0.0033377129609221453;
                  }
                }
              } else {
                result[0] += 0.010214535151773885;
              }
            } else {
              result[0] += -0.0914144531259655;
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.01634240150451749) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.06225822446579306;
              } else {
                result[0] += -0.0009106877756117191;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.006775256512495928;
              } else {
                result[0] += -0.0797383269391412;
              }
            }
          } else {
            result[0] += -0.06769518261845907;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
              result[0] += -0.04684477443804469;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += 0.03322491694288692;
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0683128784790331;
                } else {
                  result[0] += 0.0036364876332353455;
                }
              }
            }
          } else {
            result[0] += -0.035084800859578505;
          }
        } else {
          result[0] += -0.07168497596350391;
        }
      } else {
        result[0] += -0.004864046842683562;
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.0023398218770410823;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.005916242016460709;
            } else {
              result[0] += -0.09073456449068135;
            }
          } else {
            result[0] += -0.01106981070901105;
          }
        } else {
          result[0] += -0.005893452700261128;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.01658754059784981;
        } else {
          result[0] += -0.013203956079114018;
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)162.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.022279563398424754;
            } else {
              result[0] += 0.04058034719499534;
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.023486336829067783;
              } else {
                result[0] += 0.025803881536847918;
              }
            } else {
              result[0] += 0.002016736422355197;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)229.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                  result[0] += -0.013303756535312024;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.0367779115858545;
                  } else {
                    result[0] += -0.007955718442430683;
                  }
                }
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.05099444051195219;
                } else {
                  result[0] += -0.009749411233092885;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.182065486907959873) ) ) {
                    result[0] += 0.0276364422862618;
                  } else {
                    result[0] += -0.019801333746735414;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                    result[0] += 0.002838281824216689;
                  } else {
                    result[0] += -0.06405474948122963;
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)274.5000000000000568) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.71787166595459162) ) ) {
                        result[0] += 0.0036389986129592485;
                      } else {
                        result[0] += 0.035068995721208536;
                      }
                    } else {
                      result[0] += 0.05251918596595953;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                      result[0] += 0.004869136972884221;
                    } else {
                      result[0] += -0.0901920599320504;
                    }
                  }
                } else {
                  result[0] += 0.027691044094937785;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)234.5000000000000284) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.0226931390758671;
                } else {
                  result[0] += 0.0050349712493982695;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.05599822712387299;
                } else {
                  result[0] += -0.0014372536970138651;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.141444921493531162) ) ) {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.03762833148599305;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
                        result[0] += -0.04381330345063106;
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)306.5000000000000568) ) ) {
                          result[0] += -0.0023923476335833797;
                        } else {
                          result[0] += -0.019850069939834295;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
                      result[0] += 0.008864418501378304;
                    } else {
                      result[0] += -0.012124594497743758;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)298.5000000000000568) ) ) {
                    result[0] += -0.003389444478531768;
                  } else {
                    result[0] += 0.017858873885930198;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                        result[0] += -0.030240748369903692;
                      } else {
                        result[0] += 0.02777430414231742;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.017243537081191133;
                      } else {
                        result[0] += 0.017692274903466334;
                      }
                    }
                  } else {
                    result[0] += -0.03754780103161523;
                  }
                } else {
                  result[0] += -0.04859126855863935;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.14095449447632014) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
            result[0] += -0.017035074337934608;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
              result[0] += 0.006675406414226647;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.008473368267970287;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.00821172615345614;
                } else {
                  result[0] += -0.0413784140522389;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.022597199741154116;
          } else {
            result[0] += -0.07194407513284087;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += 0.06955681555464899;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
              result[0] += -0.05284090875803027;
            } else {
              result[0] += 0.031047682566260768;
            }
          } else {
            result[0] += 0.03380171181129796;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.569529533386231357) ) ) {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                result[0] += -0.023409661295563083;
              } else {
                result[0] += -0.00010698987257819328;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.024708118090790536;
              } else {
                result[0] += 0.015749396992739317;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += -0.01348704398740999;
            } else {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.0425891755247845;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.03335628993569389;
                } else {
                  result[0] += 0.02839237721775146;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.004080582187678217;
          } else {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.89009761810302912) ) ) {
                result[0] += -0.11450564996973725;
              } else {
                result[0] += -0.02983778273304613;
              }
            } else {
              result[0] += -0.020617684354163977;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.04332996349763549;
            } else {
              result[0] += 0.039052904221562726;
            }
          } else {
            result[0] += -0.02291536864662502;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9353518486022967) ) ) {
            result[0] += 0.0014109879110067549;
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.02921855689409778;
            } else {
              result[0] += -0.01663793841861758;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.650573849678039995) ) ) {
            result[0] += 0.0019448526079587896;
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)34.50000000000000711) ) ) {
              result[0] += 0.012879082866988262;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                result[0] += -0.0005171482052171819;
              } else {
                result[0] += -0.01423229822993765;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.173939466476441318) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += -0.00298557048292841;
                } else {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0716329603799545;
                    } else {
                      result[0] += -0.026973010732035452;
                    }
                  } else {
                    result[0] += -0.014421045172847292;
                  }
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                  result[0] += 0.002984454054607065;
                } else {
                  result[0] += 0.04646948383336725;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
                result[0] += 0.009882662451379736;
              } else {
                result[0] += -0.011610678128723607;
              }
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.007446195174731756;
            } else {
              result[0] += -0.036258349598984954;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.740565299987793857) ) ) {
              result[0] += 0.002566376236434273;
            } else {
              result[0] += -0.03374524768038003;
            }
          } else {
            result[0] += -0.05345084381522613;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11159896850586115) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += 0.005315442718511551;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
                  result[0] += -0.04174008096975514;
                } else {
                  result[0] += -0.002724753823428223;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
                result[0] += -0.11040910933386175;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.009856664656504744;
                } else {
                  result[0] += -0.05315435769704601;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.0038718842108364966;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                    result[0] += 0.01775199855653269;
                  } else {
                    result[0] += 0.082187201540696;
                  }
                }
              } else {
                result[0] += -0.021922767452673166;
              }
            } else {
              result[0] += 0.038373360577446024;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.01813196426354113;
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += 0.00033435476221547263;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.738182544708252841) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.0077526800859204345;
                  } else {
                    result[0] += -0.14342047593117718;
                  }
                } else {
                  result[0] += 0.08913117282228189;
                }
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.004299497380198662;
                  } else {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.12743297040008636;
                    } else {
                      result[0] += 0.02271364681592175;
                    }
                  }
                } else {
                  result[0] += 0.010006789726391206;
                }
              }
            } else {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.03090915354879467;
                } else {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.09262081478995439;
                  } else {
                    result[0] += 0.003219662434916597;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                    result[0] += 0.003882914643695993;
                  } else {
                    result[0] += 0.040624074635371474;
                  }
                } else {
                  result[0] += -0.02016901470621109;
                }
              }
            }
          } else {
            result[0] += -0.03523912192077762;
          }
        } else {
          result[0] += -0.034748259719393264;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.0009808296020481456;
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0195611339386799;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)206.5000000000000284) ) ) {
                  result[0] += -0.008173089517616331;
                } else {
                  result[0] += 0.012407692226016868;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.002253362911484653;
              } else {
                result[0] += -0.03056434487598547;
              }
            }
          } else {
            result[0] += -0.025176564333585214;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.996674776077271396) ) ) {
            result[0] += -0.058102901643777384;
          } else {
            result[0] += -0.012447631153225863;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.509355545043946201) ) ) {
            result[0] += 0.007447618788300534;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += -0.042472329485220256;
              } else {
                result[0] += 0.00873486275245344;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
                  result[0] += -0.05732243730588191;
                } else {
                  result[0] += -0.014472213046500319;
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.010825644536222812;
                } else {
                  result[0] += 0.11448317691732346;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.028780788864537128;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += 0.014130488134543696;
                } else {
                  result[0] += -0.08077985942927411;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                  result[0] += -0.017789163794625036;
                } else {
                  result[0] += -0.05592645695485102;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.04172698040351025;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.02384202723091975;
                  } else {
                    result[0] += 0.08574455681413902;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.03613711661171381;
            } else {
              result[0] += 0.014104734960159264;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.34467267990112482) ) ) {
                    result[0] += 0.012018905821811586;
                  } else {
                    result[0] += 0.06029567741135005;
                  }
                } else {
                  result[0] += 0.004510294523816577;
                }
              } else {
                result[0] += 0.0454944451463512;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.043460905197099164;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.047185741741120384;
                  } else {
                    result[0] += 0.0004062915634657268;
                  }
                }
              } else {
                result[0] += 0.015559326222659379;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.003300789164520606;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1822547912597674) ) ) {
                  result[0] += 0.005231685385665972;
                } else {
                  result[0] += -0.08650402543702862;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05938661207960886;
                } else {
                  result[0] += -0.027196510265300886;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)205.5000000000000284) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)165.5000000000000284) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.853637218475342685) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.07162382409866276;
                          } else {
                            result[0] += -0.0002400869910124359;
                          }
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.04017903669125161;
                          } else {
                            result[0] += 0.019546716819359215;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.0240028746447285;
                        } else {
                          result[0] += -0.12223824850724668;
                        }
                      }
                    } else {
                      result[0] += -0.04922789064990573;
                    }
                  } else {
                    result[0] += 0.06703522757575063;
                  }
                } else {
                  result[0] += 0.0106829646682178;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
                result[0] += 0.02749850057417082;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.424685239791871005) ) ) {
                  result[0] += -0.023980711496647582;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)275.5000000000000568) ) ) {
                    result[0] += 0.009224809503097462;
                  } else {
                    result[0] += -0.015466062504314869;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)176.5000000000000284) ) ) {
                result[0] += -0.009939889638824423;
              } else {
                result[0] += -0.02930147575094263;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03862532148767503;
            } else {
              result[0] += -0.0004340932499053193;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += -0.07035024058686536;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06523416045738831;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += -0.05710901167496583;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += 0.022994687736782617;
                  } else {
                    result[0] += -0.07988149929306573;
                  }
                }
              } else {
                result[0] += -0.07236420614100664;
              }
            }
          } else {
            result[0] += -0.0882048012687936;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.998158693313599077) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.039981246811085155;
          } else {
            result[0] += 0.003304237281597019;
          }
        } else {
          result[0] += 0.025040783721363216;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
          result[0] += -0.02972184872214641;
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.018540324388867182;
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.085941076278687412) ) ) {
              result[0] += 0.0035864677465132938;
            } else {
              result[0] += 0.06814399968109049;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)190.5000000000000284) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.023502914737000308;
            } else {
              result[0] += -0.1386988507629839;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += 0.041220341832442386;
            } else {
              result[0] += 0.00875341737733472;
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
            result[0] += 0.06507019776099825;
          } else {
            result[0] += 0.0062890276775312455;
          }
        }
      } else {
        result[0] += 0.0011982458524876644;
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
                result[0] += -0.01094197023006958;
              } else {
                result[0] += 0.00458559220799036;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.715336322784424716) ) ) {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.09678456031004518;
                  } else {
                    result[0] += -0.04093466319087186;
                  }
                } else {
                  result[0] += -0.007304420037387256;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.509355545043946201) ) ) {
                  result[0] += 0.01337057060397645;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.02244858255847642;
                  } else {
                    result[0] += -0.000717305214059679;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              result[0] += -0.031008231361859675;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.04655020203866489;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.085941076278687412) ) ) {
                      result[0] += -0.0006668674824908444;
                    } else {
                      result[0] += 0.02840788968800245;
                    }
                  }
                } else {
                  result[0] += 0.0893312749453908;
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += 0.03618498481446808;
                } else {
                  result[0] += 0.08568570053628258;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
                result[0] += 0.005463941069530735;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)175.5000000000000284) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.12190021660244173;
                      } else {
                        result[0] += -0.002237030691750958;
                      }
                    } else {
                      result[0] += -0.06680147661735235;
                    }
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.868834793567657693) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += 0.008988005013210015;
                      } else {
                        result[0] += -0.006548074624592542;
                      }
                    } else {
                      result[0] += -0.06033986147358109;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                      if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.09523129331632968;
                      } else {
                        result[0] += -0.0088228769602591;
                      }
                    } else {
                      result[0] += -0.0065601413476074335;
                    }
                  } else {
                    result[0] += -0.05792759405468443;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.040184386829192925;
              } else {
                result[0] += -0.003294801078479965;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.219399690628052646) ) ) {
              result[0] += -0.02685746307768315;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.500490188598633701) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.006315315244295967;
                        } else {
                          result[0] += -0.09208084958334402;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                          result[0] += 0.10436213280872247;
                        } else {
                          result[0] += 0.012624089811475983;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
                          result[0] += -0.07564417512116675;
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                            result[0] += -0.08228744408818672;
                          } else {
                            result[0] += -0.0036714051713155013;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.03788748160369571;
                        } else {
                          result[0] += 0.03851313641227647;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                      result[0] += 0.008366635598986141;
                    } else {
                      result[0] += 0.05127976803390101;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += 0.005476256733094619;
                  } else {
                    result[0] += 0.10751286786905628;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                  result[0] += -0.03384673603180828;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                    result[0] += -0.004015642405785167;
                  } else {
                    result[0] += 0.04897110937886405;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.09846625948342709;
        } else {
          result[0] += -0.010529483136926298;
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
        result[0] += -0.008847945768915438;
      } else {
        result[0] += -0.06411264533680248;
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.01206045709133267;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.356279611587525302) ) ) {
                result[0] += -0.02022858074481853;
              } else {
                result[0] += 0.015787504058477078;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += -0.0660870602175942;
                } else {
                  result[0] += -0.0038338276111133406;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.043387413024902788) ) ) {
                    result[0] += 0.051817748261919365;
                  } else {
                    result[0] += 0.006707386722048527;
                  }
                } else {
                  result[0] += -0.05172498410883031;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.045810627761422765;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.026927209138584252;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.009838855356815376;
                } else {
                  result[0] += 0.01831260660239255;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.08326980459162098;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    result[0] += -0.049817907462173794;
                  } else {
                    result[0] += -0.015386498502170196;
                  }
                }
              } else {
                result[0] += -0.07401894221436449;
              }
            } else {
              result[0] += 0.03047879457122444;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)283.5000000000000568) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.004039011362243032;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += 0.03980217583368814;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.249904870986938921) ) ) {
                    result[0] += 0.027266760080235958;
                  } else {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.004197731254785032;
                    } else {
                      result[0] += 0.019139847773678834;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.040213701410593244;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
              result[0] += 0.008065314403940486;
            } else {
              result[0] += -0.03543525663570183;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.542785167694092685) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              result[0] += 0.04786231699296625;
            } else {
              result[0] += -0.09378456780935746;
            }
          } else {
            result[0] += -0.06215620118720775;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)45.50000000000000711) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)33.50000000000000711) ) ) {
            result[0] += -0.0027583370642467976;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
              result[0] += -0.024209645517472005;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.70078086853027521) ) ) {
                    result[0] += -0.15223915231637808;
                  } else {
                    result[0] += -0.010757844799071612;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.47345590591430842) ) ) {
                    result[0] += 0.0492117277983483;
                  } else {
                    result[0] += -0.0227556599071443;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.007750441047503213;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += 0.064308389227346;
                    } else {
                      result[0] += 0.030515457730411245;
                    }
                  }
                } else {
                  result[0] += 0.0004342263958254653;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0008884627406243149;
            } else {
              result[0] += -0.03327789180610842;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.005355027497042508;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.681859493255617011) ) ) {
                      result[0] += -0.008064727714837145;
                    } else {
                      result[0] += -0.03965661563434454;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.288152217864991123) ) ) {
                      result[0] += -0.04538648887361332;
                    } else {
                      result[0] += 0.022303839862190825;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0046704087398325685;
                  } else {
                    result[0] += 0.06301402661400024;
                  }
                } else {
                  result[0] += -0.04301473040435846;
                }
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.07549008501016384;
              } else {
                result[0] += -0.03464356299652824;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.0694289745758555;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
            result[0] += -0.048084417746432004;
          } else {
            result[0] += -0.00237276003175689;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
      result[0] += 0.0010907256964836117;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            result[0] += 0.021779338714485896;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
              result[0] += -0.07178131923115502;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.002914144930334948;
              } else {
                result[0] += -0.09053247226981444;
              }
            }
          }
        } else {
          result[0] += 0.014958898996777763;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.269673109054566318) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.019974095157994975;
            } else {
              result[0] += -0.09161342357375306;
            }
          } else {
            result[0] += -0.057912336675831955;
          }
        } else {
          result[0] += 0.02245798949878889;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.63266015052795499) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.198464870452881303) ) ) {
                  result[0] += 0.007202490858013788;
                } else {
                  result[0] += 0.09638488191538194;
                }
              } else {
                result[0] += -0.03251122353033436;
              }
            } else {
              result[0] += 0.042805603113571206;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)166.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.04530466882345346;
                } else {
                  result[0] += -0.08032390928083877;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.010300439275882537;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.12543555892521172;
                      } else {
                        result[0] += 0.02772911893168142;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                        result[0] += 0.08222269941080683;
                      } else {
                        result[0] += -0.017408545718929247;
                      }
                    }
                  } else {
                    result[0] += 0.0644191483377573;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.826510190963745561) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                  result[0] += -0.005023290494501889;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                      result[0] += 0.07990582229323798;
                    } else {
                      result[0] += -0.03929730503499631;
                    }
                  } else {
                    result[0] += 0.13617133023103714;
                  }
                }
              } else {
                result[0] += 0.004258760911334636;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.0422482828283622;
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
                result[0] += -0.02752718667043366;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.008850729606463112;
                } else {
                  result[0] += 0.01702298351014773;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.043824925342049285;
            } else {
              result[0] += 0.005699362792550463;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)283.5000000000000568) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.655405282974244052) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.743881702423096591) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                    result[0] += 0.018090444268908552;
                  } else {
                    result[0] += -0.019908353569690673;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += 0.05965868373228658;
                  } else {
                    result[0] += 0.006759958844580276;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.249904870986938921) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.04475454387453053;
                        } else {
                          result[0] += 0.08454137657953582;
                        }
                      } else {
                        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.0019005243129370485;
                        } else {
                          result[0] += 0.10863151207215993;
                        }
                      }
                    } else {
                      result[0] += 0.0803021509584835;
                    }
                  } else {
                    result[0] += 0.04006119022275201;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                    result[0] += 0.016347016691073947;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                      result[0] += -0.0503827822100154;
                    } else {
                      result[0] += 0.0021301027252120294;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.11278031537045943;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
              result[0] += 0.018453713395639476;
            } else {
              result[0] += -0.06671141885247078;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.005797169129650549;
          } else {
            result[0] += -0.03881329801047957;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.947818994522095615) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.795762062072754794) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.00810067829638955;
                      } else {
                        result[0] += -0.022267441961692037;
                      }
                    } else {
                      result[0] += -0.0377955971524727;
                    }
                  } else {
                    result[0] += 0.016373217732737255;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)290.5000000000000568) ) ) {
                    result[0] += -0.05099320547663636;
                  } else {
                    result[0] += 0.04559296465929067;
                  }
                }
              } else {
                result[0] += 0.010280242970504276;
              }
            } else {
              result[0] += 0.04340543949478126;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.418550252914429599) ) ) {
                result[0] += 0.02247758255674335;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.007942147644191036;
                } else {
                  result[0] += 0.011389467508003976;
                }
              }
            } else {
              result[0] += -0.014352365613499472;
            }
          }
        } else {
          result[0] += -0.017465305511741604;
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.06633062260762215;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.017295712706513602;
          } else {
            result[0] += -0.08444478851541157;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.00102813945581901;
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        result[0] += -0.0013427907782985404;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
          result[0] += -0.002844896448431983;
        } else {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.10335957207224633;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.013844631274509647;
            } else {
              result[0] += -0.05734357292906143;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.367881059646607333) ) ) {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.07701695821134183;
              } else {
                result[0] += 0.018895206180890786;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.08321744710359485;
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.01060947207342473;
                  } else {
                    result[0] += -0.04085298507523592;
                  }
                } else {
                  result[0] += 0.06617538699898527;
                }
              }
            }
          } else {
            result[0] += -0.07623511117818775;
          }
        } else {
          result[0] += -0.09937609830541061;
        }
      } else {
        result[0] += 0.05062091627494303;
      }
    } else {
      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.041458208611318846;
        } else {
          result[0] += -0.0004616411355966007;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.542785167694092685) ) ) {
          result[0] += -0.012668210474783007;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
            result[0] += 0.002945886382112285;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.03700285325822031;
            } else {
              result[0] += 0.0332859676737078;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.08698434998648935;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.013345484945769604;
            } else {
              result[0] += -0.025632905955385973;
            }
          }
        } else {
          result[0] += -0.04802973692072386;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.02523542148584994;
              } else {
                result[0] += 0.09423405235052923;
              }
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += 0.005576341688142577;
                  } else {
                    result[0] += -0.03463583408274224;
                  }
                } else {
                  result[0] += 0.00951261993876969;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.040763415881967614;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += 0.01283753251758161;
                  } else {
                    result[0] += -0.033659640719764156;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += 0.06258788301634459;
            } else {
              result[0] += 0.01333115498912594;
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                result[0] += -0.007974252287396266;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += 0.16637513319599806;
                } else {
                  result[0] += 0.03459007726517564;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                  result[0] += -0.03739121391283017;
                } else {
                  result[0] += -0.10853622336575448;
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.026171795915765784;
                  } else {
                    result[0] += 0.014618396213550464;
                  }
                } else {
                  result[0] += -0.02258156013574573;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.47712564468383967) ) ) {
              result[0] += -0.0020744016179572266;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.020127415657043901) ) ) {
                            result[0] += -0.03071350037851276;
                          } else {
                            result[0] += 0.03790289634622762;
                          }
                        } else {
                          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.10716171262764794;
                          } else {
                            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                                result[0] += 0.031186888019034683;
                              } else {
                                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += -0.07093852299756204;
                                } else {
                                  result[0] += 0.001575328157730259;
                                }
                              }
                            } else {
                              result[0] += 0.04353237777667279;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.04802040262646458;
                      }
                    } else {
                      result[0] += 0.0009553766838524143;
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                            result[0] += -0.12518154336676346;
                          } else {
                            result[0] += 0.08024342800907935;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                              result[0] += -0.10553426044663;
                            } else {
                              result[0] += 0.088878831643507;
                            }
                          } else {
                            result[0] += -0.04524476822566256;
                          }
                        }
                      } else {
                        result[0] += 0.06930951067634243;
                      }
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.836270570755005771) ) ) {
                          result[0] += 0.08684068434318154;
                        } else {
                          result[0] += -0.014285167602868888;
                        }
                      } else {
                        result[0] += -0.13757189997387886;
                      }
                    }
                  }
                } else {
                  result[0] += 0.050002730979707435;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.04301172023942709;
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.002412893725042124;
                  } else {
                    result[0] += -0.037071944348891585;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
          result[0] += 0.011619146726084172;
        } else {
          result[0] += -0.02656011907341714;
        }
      } else {
        result[0] += -0.05929155237573217;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
        result[0] += -0.0004380317040031884;
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.019800814753164513;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.33581686019897639) ) ) {
                result[0] += 0.02338554727095246;
              } else {
                result[0] += -0.02483888052449769;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.026100196277206823;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.03132139044578223;
                } else {
                  result[0] += -0.09094214596530774;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.97070193290710538) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.014563590908558991;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.238486170768738237) ) ) {
                          result[0] += 0.040376345191131466;
                        } else {
                          result[0] += -0.006896792740926451;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += -0.04041405427820304;
                        } else {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                              if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
                                result[0] += -0.010337838538268968;
                              } else {
                                result[0] += -0.060246140541972616;
                              }
                            } else {
                              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += 0.02055097586995;
                              } else {
                                result[0] += -0.02703669075446922;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += 0.028176209506105616;
                            } else {
                              result[0] += -0.021556984389849973;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.03188589784652026;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.02703133207419703;
                    } else {
                      result[0] += 0.006259838847012915;
                    }
                  }
                } else {
                  result[0] += -0.021405868017632685;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.999007225036621982) ) ) {
                  result[0] += -0.061595720551717914;
                } else {
                  result[0] += -0.001389013742607256;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.799905776977539951) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                            result[0] += -0.0002604528883154048;
                          } else {
                            result[0] += -0.04800996818134737;
                          }
                        } else {
                          result[0] += -0.04925479648040283;
                        }
                      } else {
                        result[0] += -0.049613355432956266;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += -0.012660013090498932;
                      } else {
                        result[0] += -0.0629237920888406;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.03092668894127388;
                    } else {
                      result[0] += -0.018707645534425166;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                            result[0] += 0.020312821392393998;
                          } else {
                            result[0] += -0.07784941569914411;
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                                result[0] += 0.017993142576535708;
                              } else {
                                result[0] += 0.044014212285214294;
                              }
                            } else {
                              result[0] += -0.006161554080335387;
                            }
                          } else {
                            result[0] += -0.03575287539306129;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.019821283556423597;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                            result[0] += -0.10001037897512821;
                          } else {
                            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                              result[0] += 0.04658328773152057;
                            } else {
                              result[0] += -0.013581591603559774;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.04122056283890207;
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.444925546646119052) ) ) {
                        result[0] += 0.009558664805293568;
                      } else {
                        result[0] += -0.0343360327199646;
                      }
                    } else {
                      result[0] += 0.024855707540980605;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.041582148172467984;
                  } else {
                    result[0] += -0.012034851923203624;
                  }
                } else {
                  result[0] += -0.09256481295682059;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.579273939132691318) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
              result[0] += -0.007941425969949043;
            } else {
              result[0] += 0.006142173941586659;
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.032023733179057164;
            } else {
              result[0] += 0.003816570664510574;
            }
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.620046615600586826) ) ) {
            result[0] += -0.0006049753735134956;
          } else {
            result[0] += 0.013293872121031539;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
              result[0] += -0.021333167319158804;
            } else {
              result[0] += 0.0004422990641321371;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
              result[0] += -0.005535009474096377;
            } else {
              result[0] += -0.026588115865258978;
            }
          }
        } else {
          result[0] += 0.014064092574195853;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
        result[0] += -0.011186671867050187;
      } else {
        result[0] += -0.0489118457581457;
      }
    } else {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += -0.0027726826908060333;
      } else {
        result[0] += -0.03488464359281328;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
      result[0] += -9.94848757189544e-05;
    } else {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.004841358980735427;
          } else {
            result[0] += 0.04161583694803626;
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.518026351928711826) ) ) {
                  result[0] += 0.011917630932166924;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.06957334317734691;
                  } else {
                    result[0] += 0.03484819074376945;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.004248129329486589;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
                    result[0] += -0.00989188367218321;
                  } else {
                    result[0] += 0.006730206987901638;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)298.5000000000000568) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)278.5000000000000568) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                    result[0] += -0.012121163260461483;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                      result[0] += 0.009747362838655668;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.008646705218871764;
                      } else {
                        result[0] += 0.09402410446326114;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                    result[0] += -0.030196956895715367;
                  } else {
                    result[0] += 0.01697740009743866;
                  }
                }
              } else {
                result[0] += 0.02967322234541255;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.60912990570068537) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += -0.02156442058158291;
                } else {
                  result[0] += -0.052881268595750534;
                }
              } else {
                result[0] += -0.0036348201169636436;
              }
            } else {
              result[0] += 0.0023465139157637493;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.97202682495117365) ) ) {
              if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.023496989001868227;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.862926006317140448) ) ) {
                      result[0] += -0.05751469558719601;
                    } else {
                      result[0] += 0.05253063470284606;
                    }
                  } else {
                    result[0] += -0.0737879277942725;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.07606848484453155;
                  } else {
                    result[0] += -0.0006674831626141498;
                  }
                }
              }
            } else {
              result[0] += 0.04874867599800287;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.009245542410615603;
            } else {
              result[0] += -0.03914095769371104;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.006853509641223058;
              } else {
                result[0] += -0.0409584708356528;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.03762419956606877;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.040716171264650214) ) ) {
                          result[0] += 0.07061791570474321;
                        } else {
                          result[0] += -0.004345565338123458;
                        }
                      } else {
                        result[0] += -0.029289872669496193;
                      }
                    } else {
                      result[0] += -0.09564913943506387;
                    }
                  }
                } else {
                  result[0] += 0.03629988147938355;
                }
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.05031322668873886;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                          result[0] += -0.00736402511288374;
                        } else {
                          result[0] += 0.05134629353132504;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += -0.16401887034330467;
                      } else {
                        result[0] += -0.0039729601147417614;
                      }
                    }
                  } else {
                    result[0] += -0.002275708673358307;
                  }
                } else {
                  result[0] += 0.03888667351125932;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.04436639493416944;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.09129423568909054;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.03382897555068292;
                    } else {
                      result[0] += -0.03803224541951238;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                      result[0] += -0.059497139553233006;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.0006515801327412336;
                      } else {
                        result[0] += -0.08936733385790332;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.012082867763229066;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                        result[0] += 0.029983362579645335;
                      } else {
                        result[0] += 0.10745318667597255;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.0003500156356479216;
                  } else {
                    result[0] += -0.047654993023110104;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.013201527908568118;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.019845801395542678;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.11111450568821378;
                  } else {
                    result[0] += 0.02835915637753323;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
        result[0] += -0.00862880997171908;
      } else {
        result[0] += -0.036411256659634024;
      }
    } else {
      result[0] += -0.003047369295361568;
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 0.0015338142836592433;
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.590985536575318271) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04898623523194883;
                } else {
                  result[0] += 0.008583935492044769;
                }
              } else {
                result[0] += -0.02292078483885628;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.05447296620533957;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)248.5000000000000284) ) ) {
                      result[0] += -0.005535731706431579;
                    } else {
                      result[0] += 0.012874135463140236;
                    }
                  }
                } else {
                  result[0] += 0.015682040839965255;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.01273877027738634;
                } else {
                  result[0] += 0.04355557092960917;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.594915628433228427) ) ) {
                result[0] += -0.00878885381950264;
              } else {
                result[0] += -0.03995412359371991;
              }
            } else {
              result[0] += 0.004643780936926628;
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79835033416748225) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                  result[0] += -0.018774362011448605;
                } else {
                  result[0] += 0.007220418658322004;
                }
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.014214329584390631;
                } else {
                  result[0] += -0.003629728916667962;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                result[0] += 0.0044614969340307795;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.0029680795516207757;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.04675480574550464;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.03390304693851243;
                    } else {
                      result[0] += 0.002307580199766232;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)221.5000000000000284) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.003792329807871522;
              } else {
                result[0] += -0.027729895216110762;
              }
            } else {
              result[0] += -0.03325698996615419;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.04942942390887755;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.192109584808350498) ) ) {
                  result[0] += -0.0710019470922371;
                } else {
                  result[0] += 0.009873054208246487;
                }
              } else {
                result[0] += -0.004142293489419624;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.018807071436722913;
                } else {
                  result[0] += -0.1122181289865025;
                }
              } else {
                result[0] += 0.017046368173242992;
              }
            } else {
              result[0] += 0.02347223817802375;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.570956468582154208) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)256.5000000000000568) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.01906846887111917;
                  } else {
                    result[0] += -0.06590635396915566;
                  }
                } else {
                  result[0] += 0.021989859780764634;
                }
              } else {
                result[0] += 0.002201352854879387;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                result[0] += 0.012052802371707014;
              } else {
                result[0] += -0.012639203631085581;
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)128.5000000000000284) ) ) {
              result[0] += -0.0006044352125729336;
            } else {
              result[0] += -0.021344904716252583;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
      result[0] += -8.963451069656288e-05;
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
          result[0] += 0.04024973859657932;
        } else {
          result[0] += 0.01319199294105048;
        }
      } else {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.013416716766423762;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.998158693313599077) ) ) {
                  result[0] += -0.05114355999518419;
                } else {
                  result[0] += 0.0388526150293372;
                }
              }
            } else {
              result[0] += -0.020034730113024733;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                  result[0] += 0.005049245247646033;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.549732685089113104) ) ) {
                    result[0] += -0.07671519613169221;
                  } else {
                    result[0] += 0.017523089819393958;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                  result[0] += 0.035569983460833336;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.03964083772902395;
                    } else {
                      result[0] += 0.026932491717178;
                    }
                  } else {
                    result[0] += -0.005380904349544971;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                      result[0] += -0.05163356852488491;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.01141517092430857;
                      } else {
                        result[0] += -0.07981888629583045;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.004120254654247923;
                    } else {
                      result[0] += 0.045284970122175416;
                    }
                  }
                } else {
                  result[0] += -0.027081640541901694;
                }
              } else {
                result[0] += 0.05218039979814581;
              }
            }
          }
        } else {
          result[0] += -0.020029833074201295;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)323.5000000000000568) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.0016093561002821404;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.665476083755494052) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.828941345214844638) ) ) {
              result[0] += -0.0007673158349464682;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += -0.010350374334745564;
              } else {
                result[0] += -0.06898048536088193;
              }
            }
          } else {
            result[0] += 0.0060599689478278455;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.01105627422885437;
          } else {
            result[0] += 0.002346157807206655;
          }
        } else {
          result[0] += 0.003080837443444635;
        }
      }
    } else {
      result[0] += -0.024789349521973106;
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
      if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)268.5000000000000568) ) ) {
                result[0] += -0.0001263253075976853;
              } else {
                result[0] += 0.041748237007388146;
              }
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                  result[0] += 0.00046534019153936766;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)312.5000000000000568) ) ) {
                    result[0] += 0.025128929879574758;
                  } else {
                    result[0] += -0.027477029197096565;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.500490188598633701) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += -0.0703552236834143;
                    } else {
                      result[0] += -0.010305117764331094;
                    }
                  } else {
                    result[0] += 0.06665552073708651;
                  }
                } else {
                  result[0] += -0.07621390229378845;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)300.5000000000000568) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                    result[0] += -0.05451911232608919;
                  } else {
                    result[0] += -0.12230241987759219;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                      result[0] += -0.024933575037293886;
                    } else {
                      result[0] += 0.04722526444316867;
                    }
                  } else {
                    result[0] += -0.10173306516490051;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                  result[0] += -0.04558007237017558;
                } else {
                  result[0] += 0.047737231276034076;
                }
              }
            } else {
              result[0] += 0.013356519647299487;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9353518486022967) ) ) {
              result[0] += -0.010714065519975854;
            } else {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.020259627365661184;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                  result[0] += 0.04887473558979763;
                } else {
                  result[0] += -0.10389707438946437;
                }
              }
            }
          } else {
            result[0] += 0.02540359326629837;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.939840793609620917) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.0022296235495313024;
            } else {
              result[0] += -0.015972992144968876;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.0022715025482949957;
              } else {
                result[0] += -0.025957837746828156;
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += 0.052201910541301466;
              } else {
                result[0] += -0.03029239707147697;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            result[0] += 0.012725377554440512;
          } else {
            result[0] += -0.0347441073879672;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)171.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                    result[0] += -0.03770301030152803;
                  } else {
                    result[0] += 0.00039776227951615774;
                  }
                } else {
                  result[0] += 0.011022991441160963;
                }
              } else {
                result[0] += -0.0801233903829553;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                result[0] += 0.0074710184048022085;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.014303780366494934;
                } else {
                  result[0] += 0.08038934444111061;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.002084029180734025;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12938737869262873) ) ) {
                    result[0] += -0.015623499833660069;
                  } else {
                    result[0] += 0.027936214440637465;
                  }
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.048059246328232996;
                  } else {
                    result[0] += -0.009225190574416068;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.26837396621704279) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.09863122819199592;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)123.5000000000000142) ) ) {
                        result[0] += -0.031503717702083225;
                      } else {
                        result[0] += 0.06783832728957442;
                      }
                    }
                  } else {
                    result[0] += 0.029710428671302393;
                  }
                } else {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.03175867049183425;
                  } else {
                    result[0] += -0.01814036925339595;
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
                      result[0] += 0.04201372942994175;
                    } else {
                      result[0] += -0.06197366241800395;
                    }
                  } else {
                    result[0] += -0.006484539189742499;
                  }
                } else {
                  result[0] += 0.04583222574304192;
                }
              }
            }
          }
        } else {
          result[0] += 0.0030028848397191717;
        }
      } else {
        result[0] += -0.011312311420056155;
      }
    }
  }
  if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)306.5000000000000568) ) ) {
        result[0] += 0.0022218790317600754;
      } else {
        result[0] += -0.0173626503351911;
      }
    } else {
      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)6144.000000000000909) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.509355545043946201) ) ) {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                result[0] += -0.005926750140404242;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                  if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.04671703909247393;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0005055765438821222;
                    } else {
                      result[0] += 0.04619798755375745;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)72.50000000000001421) ) ) {
                    result[0] += 0.001638495090548614;
                  } else {
                    result[0] += 0.055255850795054774;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.76779222488403498) ) ) {
                result[0] += 0.002900691215794294;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.029665455951693693;
                  } else {
                    result[0] += -0.01237615655445509;
                  }
                } else {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.06876629750989836;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                        result[0] += 0.08032340992397007;
                      } else {
                        result[0] += -0.012041584836948586;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += 0.009690510793308734;
                    } else {
                      result[0] += -0.05870710126387812;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.382196187973023349) ) ) {
                    result[0] += -0.009328351034006482;
                  } else {
                    result[0] += 0.07117576453335858;
                  }
                } else {
                  result[0] += 0.07592722767151008;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.003975649929528834;
                  } else {
                    result[0] += -0.02946852279473416;
                  }
                } else {
                  result[0] += -0.025677315317810392;
                }
              }
            } else {
              result[0] += -0.0006854764053541154;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.97070193290710538) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.843275547027588779) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.02108760235767674;
                  } else {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.04066844141723248;
                    } else {
                      result[0] += -0.08441040032104802;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.10112211135698053;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
                        result[0] += 0.005489490445620311;
                      } else {
                        result[0] += -0.04497139921401164;
                      }
                    } else {
                      result[0] += 0.029174648504215785;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                      result[0] += -0.02644004032356087;
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)65.50000000000001421) ) ) {
                          if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.04833994279451746;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10559129714965998) ) ) {
                              result[0] += -0.03494563681832233;
                            } else {
                              result[0] += 0.04870066770781667;
                            }
                          }
                        } else {
                          result[0] += -0.01459942489988758;
                        }
                      } else {
                        result[0] += 0.04647869804753006;
                      }
                    }
                  } else {
                    result[0] += -0.01727245604267486;
                  }
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.024513424160302662;
                  } else {
                    result[0] += 0.11264931071803097;
                  }
                }
              }
            } else {
              result[0] += -0.05651194445178192;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.005933611050916565;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
                result[0] += -0.0006030857096655628;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += 0.007387177501669723;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.539540290832521308) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.021532390970715243;
                      } else {
                        result[0] += 0.013561654965778;
                      }
                    } else {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
                          result[0] += -0.019946406402854278;
                        } else {
                          result[0] += -0.05745904616369797;
                        }
                      } else {
                        result[0] += -0.05119285617397789;
                      }
                    }
                  } else {
                    result[0] += -0.0024004823172643283;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.02016203269413377;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += -0.0031203843683418665;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)123.5000000000000142) ) ) {
              result[0] += -0.10186522676897757;
            } else {
              result[0] += 0.004374943532549576;
            }
          } else {
            result[0] += -0.04346280139671372;
          }
        }
      } else {
        result[0] += 0.004679758556684799;
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.005392313163043987;
          } else {
            result[0] += 0.01949671218793425;
          }
        } else {
          result[0] += 0.004105262318613975;
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)225.5000000000000284) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.00426427604256854;
          } else {
            result[0] += 0.056332553013648735;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.549732685089113104) ) ) {
              result[0] += 0.02384017551666441;
            } else {
              result[0] += -0.05291806395346227;
            }
          } else {
            result[0] += -0.0394343595573847;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.051912069320679599) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)227.5000000000000284) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.0027363865152619277;
                  } else {
                    result[0] += -0.03921240684434177;
                  }
                } else {
                  result[0] += -0.007611845428982921;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
                  result[0] += -0.03522965902066984;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)89.50000000000001421) ) ) {
                    result[0] += 0.02043332448919112;
                  } else {
                    result[0] += -0.006365657567305293;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)296.5000000000000568) ) ) {
                  result[0] += 0.016929935202431672;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.03733885618847137;
                  } else {
                    result[0] += -0.008998437123637324;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.024774140833536313;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.059174763953152;
                  } else {
                    result[0] += 0.002470538887392774;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)268.5000000000000568) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.971427202224732333) ) ) {
                  result[0] += -0.07696105754775895;
                } else {
                  result[0] += -0.028827967121379035;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.06059598667533137;
                } else {
                  result[0] += -0.0022950620577330606;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.008326210736908705;
              } else {
                result[0] += -0.0399853758391274;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)27.50000000000000355) ) ) {
            result[0] += -0.009373515240832539;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
                result[0] += 0.01567498208393554;
              } else {
                result[0] += -0.006419657848812552;
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.002814627950533652;
                  } else {
                    result[0] += 0.021884354626552503;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.013602789371774749;
                    } else {
                      result[0] += -0.011664785572286757;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.04830147069136931;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                        result[0] += 0.040661368837466856;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.206374883651734287) ) ) {
                          result[0] += 0.0016167520050783047;
                        } else {
                          result[0] += -0.03597297747813914;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0004893549633378671;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0027084756050039004;
              } else {
                result[0] += -0.015053606619867302;
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.014657533904194019;
              } else {
                result[0] += -0.008184364037613115;
              }
            }
          } else {
            if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.01170722712716421;
                  } else {
                    result[0] += 0.03637535946933775;
                  }
                } else {
                  result[0] += -0.011854702568049724;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)229.5000000000000284) ) ) {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.022556450114400554;
                    } else {
                      result[0] += -0.05178956130959807;
                    }
                  } else {
                    result[0] += -0.007144250514826131;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                    result[0] += 0.01877667364341333;
                  } else {
                    result[0] += -0.09802497063565438;
                  }
                }
              }
            } else {
              result[0] += 0.016977618076035157;
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += -0.0031477586215221593;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.035428574606448125;
                } else {
                  result[0] += -0.06719737820214776;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)220.5000000000000284) ) ) {
                  result[0] += 0.014653784805342194;
                } else {
                  result[0] += -0.02678981069741747;
                }
              } else {
                result[0] += -0.08754701592479769;
              }
            }
          } else {
            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += -0.03035306335617798;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)168.5000000000000284) ) ) {
                    result[0] += 0.03826168106134017;
                  } else {
                    result[0] += 0.005735972317179974;
                  }
                }
              } else {
                result[0] += -0.036120834136665246;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.05479200500527662;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.038255839289371076;
                  } else {
                    result[0] += 0.01392546690773677;
                  }
                } else {
                  result[0] += 0.008484965599435072;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.0017213267880722687;
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
        result[0] += -0.001666944383105615;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.030344725804248986;
          } else {
            result[0] += 0.00408712400663421;
          }
        } else {
          result[0] += -0.04593956861744887;
        }
      }
    } else {
      result[0] += 0.022520108608555772;
    }
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    result[0] += -0.0011096398651964715;
  } else {
    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.8192477226257342) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.006705003476938947;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.269673109054566318) ) ) {
                    result[0] += 0.030845370913497795;
                  } else {
                    result[0] += 0.09450378757505501;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                    result[0] += -0.06744656339798741;
                  } else {
                    result[0] += 0.015176143699541922;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                  result[0] += 0.03021743936206285;
                } else {
                  result[0] += -0.0016858657714519901;
                }
              } else {
                result[0] += -0.028779185146057706;
              }
            }
          } else {
            result[0] += 0.003131096938592023;
          }
        } else {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
              result[0] += -0.01406559308352237;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.08630940172847214;
              } else {
                result[0] += -0.01444657906101908;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.06785706676807533;
                } else {
                  result[0] += -0.03191590369297805;
                }
              } else {
                result[0] += 0.013675152064584556;
              }
            } else {
              result[0] += 0.027196958368333718;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.001351356506349433) ) ) {
          result[0] += -0.03888656844822261;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += -0.018770166521879034;
              } else {
                result[0] += 0.01094519131941537;
              }
            } else {
              result[0] += 0.09392938300496124;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.34467267990112482) ) ) {
              result[0] += -0.04755652711207949;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.0336212790395321;
              } else {
                result[0] += -0.027426877022306985;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
            result[0] += -0.0575504786304376;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.007621391801669046;
            } else {
              result[0] += -0.06849105200715583;
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
            result[0] += 0.0020368679859003778;
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                  result[0] += 0.02526791148650988;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.022709695801723856;
                    } else {
                      result[0] += -0.11194121141117384;
                    }
                  } else {
                    result[0] += 0.0007826025609246222;
                  }
                }
              } else {
                result[0] += -0.04962370477964518;
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                  result[0] += 0.010953624550632856;
                } else {
                  result[0] += -0.04272335468099833;
                }
              } else {
                if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)295.5000000000000568) ) ) {
                    result[0] += 0.03116230411345551;
                  } else {
                    result[0] += -0.008976112496278865;
                  }
                } else {
                  result[0] += -0.008607335118389639;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.402451276779175693) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)299.5000000000000568) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.0004030482034089753;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.02389638106576418;
                } else {
                  result[0] += -0.07016274563730748;
                }
              }
            } else {
              result[0] += -0.037652339854360874;
            }
          } else {
            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.05772322853915934;
            } else {
              result[0] += -0.02676976272284494;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
            result[0] += 0.009140827974290491;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.000786488303079225;
                } else {
                  result[0] += 0.03840406784413611;
                }
              } else {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.145964622497559482) ) ) {
                    result[0] += -0.1879709706103552;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.028073487368639984;
                        } else {
                          result[0] += -0.010675470550483682;
                        }
                      } else {
                        result[0] += -0.06492773725017523;
                      }
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.039256066829071706;
                      } else {
                        result[0] += -0.021425091188885567;
                      }
                    }
                  }
                } else {
                  result[0] += -0.030610980740916328;
                }
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.0650905688862839;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)271.5000000000000568) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06165723427342773;
                    } else {
                      if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.03681254852009351;
                        } else {
                          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                              result[0] += 0.06369141908745314;
                            } else {
                              result[0] += 0.0034282579637300407;
                            }
                          } else {
                            result[0] += -0.019744444504307894;
                          }
                        }
                      } else {
                        result[0] += -0.04042461579284699;
                      }
                    }
                  } else {
                    result[0] += 0.013216091131851175;
                  }
                } else {
                  result[0] += -0.05114994106222068;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += 0.027579624121394943;
                } else {
                  result[0] += 0.004405009400620478;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += -0.12037885222126549;
                } else {
                  result[0] += -0.010249120230346003;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.002659109212357784;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.06905909601263785;
                    } else {
                      result[0] += -0.027783321741120394;
                    }
                  }
                } else {
                  result[0] += -0.007955759884645105;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  result[0] += -0.038987967672378526;
                } else {
                  result[0] += 0.004354036533429978;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.334978580474854404) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.00941239427541811;
              } else {
                result[0] += -0.06010373552307746;
              }
            } else {
              result[0] += 0.008224140822186045;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.01099915040847451;
          } else {
            result[0] += 0.0005539132682663978;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
                  result[0] += -0.03116133897515764;
                } else {
                  result[0] += -0.00014368777892002286;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                  result[0] += 0.014995613638184078;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02738215401282902;
                  } else {
                    result[0] += 0.018120677913749352;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.03332386552777557;
                } else {
                  result[0] += -0.017629150306247978;
                }
              } else {
                result[0] += 0.022023151711381972;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.03161643382493238;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.017635883988922282;
                    } else {
                      result[0] += 0.018095975749828266;
                    }
                  }
                } else {
                  result[0] += -0.038173316437267964;
                }
              } else {
                result[0] += 0.025372945382220843;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.553655147552491123) ) ) {
                    result[0] += -0.054945282283690104;
                  } else {
                    result[0] += -0.01678775669690105;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.03447722686500751;
                    } else {
                      result[0] += 0.015356709169072866;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                      result[0] += -0.051129085815408794;
                    } else {
                      result[0] += 0.020480481556039983;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
                      result[0] += 0.011133117176928348;
                    } else {
                      result[0] += -0.04834301085408249;
                    }
                  } else {
                    result[0] += -0.02643292624231949;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.016867923945465567;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.007654078270416609;
                          } else {
                            result[0] += -0.04879939739717179;
                          }
                        }
                      } else {
                        result[0] += 0.030540393466991524;
                      }
                    } else {
                      if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.026234860127397442;
                      } else {
                        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.053893236581744865;
                        } else {
                          result[0] += 0.004470743353473284;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.032024579419937776;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                        result[0] += 0.011403913138648795;
                      } else {
                        result[0] += -0.07392783758442605;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.030212150978060865;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += -0.007610391677263834;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                    result[0] += 0.061718089209853534;
                  } else {
                    result[0] += 0.0009383646893435188;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                result[0] += -0.06254641096608043;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.791641235351563388) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    result[0] += -0.0017582980175509387;
                  } else {
                    result[0] += -0.0856671314032129;
                  }
                } else {
                  result[0] += -0.055438155731093786;
                }
              }
            }
          } else {
            result[0] += 0.036796920219972586;
          }
        }
      }
    } else {
      result[0] += -0.006644409262256172;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.06073126209603372;
        } else {
          result[0] += -0.0796088076666247;
        }
      } else {
        result[0] += 0.0007874516053469968;
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += -0.002687792560353284;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          result[0] += -0.025468387155422652;
        } else {
          result[0] += 0.0115083198824934;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.001377112017260522;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += 0.02480300260218653;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)63.50000000000000711) ) ) {
                    result[0] += -0.048335576669533024;
                  } else {
                    result[0] += -0.01402621361859239;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.020414461700950626;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                  result[0] += -0.00029769262537068396;
                } else {
                  result[0] += 0.0302992530606328;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
              result[0] += -0.06141807517890202;
            } else {
              result[0] += -0.013458943281999617;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
            result[0] += 0.007181565666674224;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.006022582967549227;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.021655528147013017;
                } else {
                  result[0] += -0.08023594483953321;
                }
              } else {
                result[0] += 0.0019780923817578006;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0067366826941867705;
              } else {
                result[0] += -0.02294526623607708;
              }
            } else {
              result[0] += -0.06363886179690346;
            }
          } else {
            result[0] += 0.005267463223485081;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.012588831962598449;
                    } else {
                      result[0] += 0.0793837965461418;
                    }
                  } else {
                    result[0] += -0.045532440165471394;
                  }
                } else {
                  result[0] += -0.05427943184770262;
                }
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.0730858414618351;
                  } else {
                    result[0] += 0.004721221154689475;
                  }
                } else {
                  result[0] += -0.03579425552117062;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                result[0] += 0.044592529645619126;
              } else {
                result[0] += -0.009116901262589485;
              }
            }
          } else {
            result[0] += 0.03608876417546413;
          }
        }
      }
    } else {
      result[0] += -0.0060863178865291365;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.001843395139937584;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += -0.0359875114416522;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += -0.01029725710751515;
                } else {
                  result[0] += 0.023362163176919193;
                }
              }
            } else {
              result[0] += -0.06107113140011066;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
              result[0] += 0.01942961240470805;
            } else {
              result[0] += -0.06819449343772326;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.005186327848381394;
          } else {
            result[0] += -0.01997704698089849;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.004337311289279223;
          } else {
            result[0] += -0.04932017952853437;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.182065486907959873) ) ) {
              result[0] += -0.006532528073132643;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.03020322992319905;
                } else {
                  result[0] += -0.07573538879333754;
                }
              } else {
                result[0] += 0.01299136507386361;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)274.5000000000000568) ) ) {
                    result[0] += 0.03209392526783896;
                  } else {
                    result[0] += 0.0010923759678070322;
                  }
                } else {
                  result[0] += -0.01795642150225461;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)280.5000000000000568) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.550365447998047763) ) ) {
                      result[0] += -0.0010997763761725757;
                    } else {
                      result[0] += -0.1305250956797343;
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.540307998657227451) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.393745899200439897) ) ) {
                        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.04600650610971768;
                        } else {
                          result[0] += 0.04760031575788275;
                        }
                      } else {
                        result[0] += 0.12086042479308304;
                      }
                    } else {
                      result[0] += 0.11970587823237594;
                    }
                  }
                } else {
                  result[0] += 0.03646187637982064;
                }
              }
            } else {
              result[0] += -0.01453769450871337;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.05057103449703331;
                } else {
                  result[0] += 0.007550472643485336;
                }
              } else {
                result[0] += -0.040677422587942214;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.048941143804685394;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.016663898737874928;
                } else {
                  result[0] += 0.018913730574014926;
                }
              }
            }
          } else {
            result[0] += -0.06920255308518886;
          }
        } else {
          result[0] += 0.010501584340638742;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
            result[0] += 7.038774523837827e-05;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)279.5000000000000568) ) ) {
              result[0] += -0.04336057799581524;
            } else {
              result[0] += 0.009216369425814955;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.03648824308837015;
              } else {
                if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.003970768201849647;
                } else {
                  result[0] += -0.025492880634538757;
                }
              }
            } else {
              result[0] += -0.03616725315940055;
            }
          } else {
            result[0] += -0.06350993737057282;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.780892848968506748) ) ) {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.055165391262999235;
              } else {
                result[0] += 0.015389837270182153;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)71.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                  result[0] += -0.016852560603139964;
                } else {
                  result[0] += 0.040902547281379584;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.619406223297120029) ) ) {
                  result[0] += 0.0033699129683079396;
                } else {
                  result[0] += -0.03244533724611794;
                }
              }
            }
          } else {
            result[0] += 0.002176985151989586;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
            result[0] += 0.005612950606226377;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
              result[0] += -0.02794078739149102;
            } else {
              result[0] += -0.005040762988508744;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.006156463253856902;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
                    result[0] += -0.025084022425011294;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)251.5000000000000284) ) ) {
                      result[0] += -0.009461866986458345;
                    } else {
                      result[0] += 0.025191078186995505;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.061806882885528894;
                  } else {
                    result[0] += -0.017877174957919382;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)59.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += -0.0020893120794616706;
                    } else {
                      result[0] += 0.03356208979640833;
                    }
                  } else {
                    result[0] += -0.015204220075650221;
                  }
                } else {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.024137764928768783;
                  } else {
                    result[0] += 0.004328120008482565;
                  }
                }
              } else {
                result[0] += 0.008669448501391836;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.01802267665105334;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.031989184411316145;
                      } else {
                        result[0] += 0.0024165760043229306;
                      }
                    } else {
                      result[0] += 0.016238836149601237;
                    }
                  }
                } else {
                  result[0] += 0.025701985933228727;
                }
              } else {
                result[0] += -0.0035605132570812967;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.027329830368842012;
                  } else {
                    result[0] += 0.010685792590383748;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -3.2610547973071623e-05;
                  } else {
                    result[0] += 0.026266694595862634;
                  }
                }
              } else {
                result[0] += -0.019659200916923997;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                result[0] += -0.0022545478722251528;
              } else {
                result[0] += -0.044116286001877884;
              }
            } else {
              result[0] += 0.0071842804386278965;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67046499252319514) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.034718335546242415;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)219.5000000000000284) ) ) {
                  result[0] += 0.018223927983167198;
                } else {
                  result[0] += -0.0034433910035396582;
                }
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.001780263530003658;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.001923319443626062;
                  } else {
                    result[0] += 0.0822339864816309;
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                    result[0] += -0.07212692944433774;
                  } else {
                    result[0] += -0.01996414637866073;
                  }
                } else {
                  result[0] += 0.0007436212463116698;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.029462120588544857;
      }
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.085941076278687412) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        result[0] += -0.02089365136839763;
      } else {
        result[0] += -0.0003623400364267438;
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
        result[0] += -0.0038768613401370697;
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              result[0] += -0.00783226376760171;
            } else {
              result[0] += -0.04631938554421537;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.08063544899712458;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)67.50000000000001421) ) ) {
                result[0] += -0.012720681316983763;
              } else {
                result[0] += -0.05326008642032262;
              }
            }
          }
        } else {
          result[0] += 0.00428802771179219;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08776257447903549;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.242078304290772373) ) ) {
                  result[0] += -0.031387274046630026;
                } else {
                  result[0] += 0.10595273488667473;
                }
              }
            } else {
              result[0] += 0.04049620527661363;
            }
          } else {
            result[0] += -0.051488079310557655;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += 0.011681287309305503;
          } else {
            result[0] += -0.008411074147417357;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
          result[0] += -0.008914331578972534;
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.08093467229445943;
          } else {
            result[0] += -0.03447245205161187;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.356279611587525302) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.09660074570073013;
              } else {
                result[0] += -0.009846143548206947;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
                result[0] += -0.006290366255725283;
              } else {
                result[0] += 0.0256854377017251;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.051747083663941318) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.310776710510254794) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                    result[0] += -0.043989870309369676;
                  } else {
                    result[0] += 0.003747867578551595;
                  }
                } else {
                  result[0] += -0.04218400901993572;
                }
              } else {
                result[0] += -0.03625519974811531;
              }
            } else {
              result[0] += 0.0285161152253967;
            }
          }
        } else {
          result[0] += -0.010157138861893983;
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.023858785629273349) ) ) {
              result[0] += -0.015714524434422086;
            } else {
              result[0] += 0.06803908663685168;
            }
          } else {
            result[0] += -0.03486502395719908;
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.633862972259523261) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += 0.01646301244684151;
            } else {
              result[0] += -0.008717401513005396;
            }
          } else {
            result[0] += -0.031116538967972552;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.00109517450797682;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.43260431289673029) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
                result[0] += 0.0007012516568014481;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.1215934568049585;
                } else {
                  result[0] += -0.027308197278565496;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                result[0] += -0.004142298405603948;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.030447461685042438;
                } else {
                  if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.12696700552159945;
                  } else {
                    result[0] += 0.02968175295545831;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.980170249938965732) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.01242191393577709;
              } else {
                result[0] += -0.08381353402832076;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)104.5000000000000142) ) ) {
                result[0] += -0.0702692722068928;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.56833124160766779) ) ) {
                    result[0] += 0.0835295600927996;
                  } else {
                    result[0] += -0.06106810908692332;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                    result[0] += -0.02661024550394144;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.94140863418579279) ) ) {
                        result[0] += -0.0936141794568601;
                      } else {
                        result[0] += 0.05579615274153487;
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.06999168879293007;
                      } else {
                        result[0] += -0.09700891747055912;
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.000997530680081385;
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += 0.004705840921161568;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.019209957526680616;
              } else {
                result[0] += -0.017521662569367685;
              }
            } else {
              result[0] += -0.05589390710131908;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
              result[0] += -0.022375685949183852;
            } else {
              result[0] += -0.05763308824111496;
            }
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)195.5000000000000284) ) ) {
                result[0] += 0.05933434080757991;
              } else {
                result[0] += -0.019535241163891853;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.0005561198162311277;
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
                      result[0] += 0.028885375821158094;
                    } else {
                      result[0] += -0.03679562899077127;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                      result[0] += 0.0058147113570523835;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.08061936928215621;
                      } else {
                        result[0] += -0.014122058677335568;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.1113968024966148;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.01634240150451749) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.0019814515316930958;
              } else {
                result[0] += -0.061431235421120806;
              }
            } else {
              result[0] += -0.061517002046665485;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.085941076278687412) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += 0.0175380331256017;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.964135169982911044) ) ) {
              result[0] += -0.013309541828824979;
            } else {
              result[0] += -0.061290710659205244;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.145964622497559482) ) ) {
            result[0] += -0.06946699354343039;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
              result[0] += -0.05422373927334798;
            } else {
              result[0] += 0.0585383130093437;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                result[0] += -0.002142098432895584;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)45.50000000000000711) ) ) {
                  result[0] += -0.07030881201173471;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.05427197654320537;
                  } else {
                    result[0] += -0.014816531017438062;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)67.50000000000001421) ) ) {
                    result[0] += 0.024889044346070414;
                  } else {
                    result[0] += -0.06861500360370879;
                  }
                } else {
                  result[0] += -0.0076097585581964755;
                }
              } else {
                result[0] += 0.018330540727355824;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.780892848968506748) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.579839229583741123) ) ) {
                  result[0] += 0.06953836104951643;
                } else {
                  result[0] += 0.0050900292498110115;
                }
              } else {
                result[0] += 0.023388953376819677;
              }
            } else {
              result[0] += -0.02193747443932297;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += -0.0009083724916531141;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                result[0] += -0.06570791774534264;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)34.50000000000000711) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.16158268946978852;
                    } else {
                      result[0] += -0.018847644093660867;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
                        result[0] += 0.06631014931805745;
                      } else {
                        result[0] += -0.05498110430486504;
                      }
                    } else {
                      result[0] += -0.09589197687621134;
                    }
                  }
                } else {
                  result[0] += -0.03828619589046734;
                }
              }
            } else {
              result[0] += -0.0663242419357448;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
            result[0] += 0.0059957527130916925;
          } else {
            result[0] += 0.04923564784816639;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)20.50000000000000355) ) ) {
              result[0] += -0.02692094665509583;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.166635274887085849) ) ) {
                result[0] += -0.0414491328555839;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.966066360473633701) ) ) {
                    result[0] += 0.12006812975268161;
                  } else {
                    result[0] += 0.03184407508119004;
                  }
                } else {
                  result[0] += -0.0661283133429759;
                }
              }
            }
          } else {
            result[0] += -0.022164281412926867;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.65824460983276545) ) ) {
                result[0] += 0.04117517970956515;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.791641235351563388) ) ) {
                  result[0] += 0.13398126924055054;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.07443598301051328;
                    } else {
                      result[0] += 0.041820346329152505;
                    }
                  } else {
                    result[0] += 0.02065599405831113;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.009036093787840573;
              } else {
                if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.10031178496026594;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += -0.13262062486052392;
                  } else {
                    result[0] += -0.05133482378164572;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.08325510823040583;
            } else {
              result[0] += -0.026196821297312906;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.002690115746196331;
            } else {
              result[0] += -0.036875360748513795;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
              result[0] += 0.007989316957862388;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.029342823307809376;
              } else {
                result[0] += -0.003920878498657643;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
      result[0] += 0.0008074357487866547;
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.542785167694092685) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)16.50000000000000355) ) ) {
          result[0] += -0.09747374135653553;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
            result[0] += 0.07658183413453883;
          } else {
            result[0] += 0.025277584647780682;
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += 0.07435323749329063;
              } else {
                if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.04024787064680226;
                } else {
                  result[0] += 0.026476861640797877;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.149111986160279208) ) ) {
                result[0] += -0.02792434619972871;
              } else {
                result[0] += -0.08116451091843471;
              }
            }
          } else {
            result[0] += -0.058988896595638066;
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.01783710970980157;
          } else {
            result[0] += 0.08658659188177292;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.1060287458552735;
    } else {
      if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += -0.0006107850203139319;
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += 0.03335044235578857;
                  } else {
                    result[0] += -0.02723925746797909;
                  }
                } else {
                  result[0] += -0.0070958946894296195;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += 0.08142467876284123;
                } else {
                  result[0] += 0.004760978581147116;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.00673267476680928;
                } else {
                  result[0] += -0.0462961612327092;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06317750118019913;
                  } else {
                    result[0] += -9.865144715116507e-06;
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.005282842141283261;
                  } else {
                    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06987085650842444;
                    } else {
                      result[0] += 0.004579490437836377;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
              result[0] += 0.00034448880336532907;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.0213299722297973;
              } else {
                result[0] += -0.017036559425729726;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                result[0] += -0.009841025667637818;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.01965142437668194;
                  } else {
                    result[0] += -0.012261845971641892;
                  }
                } else {
                  result[0] += 0.0027650146921374616;
                }
              }
            } else {
              if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.03976471260256096;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.003383325822762214;
                  } else {
                    result[0] += -0.06413675179334132;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.007413959972605236;
                  } else {
                    result[0] += 0.03475184499495227;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += 0.013386087623905427;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.971427202224732333) ) ) {
                  result[0] += 0.0018557645126617154;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                        result[0] += -0.03737223601311195;
                      } else {
                        result[0] += 0.00018856361924877066;
                      }
                    } else {
                      result[0] += -0.04853877819078458;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
                          result[0] += 0.06351688948975538;
                        } else {
                          result[0] += -0.012234023097083883;
                        }
                      } else {
                        result[0] += -0.049260089623753334;
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.019854302958089945;
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
                          result[0] += -0.027638724948341676;
                        } else {
                          result[0] += 0.020474844400109292;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  result[0] += -0.02084575288682676;
                } else {
                  result[0] += 0.015092328100062155;
                }
              } else {
                result[0] += -0.03408428211194635;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.01544202675773266;
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.032969291330204556;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.033077404488398514;
            } else {
              result[0] += 0.0026756103397418854;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.05460885857423047;
          } else {
            result[0] += 0.004512349343617824;
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.241249561309815341) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.404924631118775302) ) ) {
                    result[0] += -0.0026595091381603625;
                  } else {
                    result[0] += -0.06072410956038883;
                  }
                } else {
                  result[0] += 0.029552629981521528;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.048973218520122534;
                } else {
                  result[0] += 0.000801592851804425;
                }
              }
            } else {
              result[0] += -0.027671255498302114;
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.40819787979126154) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.040422139036562214;
                    } else {
                      result[0] += -0.03803973747130337;
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.1027499245209984;
                    } else {
                      result[0] += 0.008314398186534492;
                    }
                  }
                } else {
                  result[0] += -0.053806230974257185;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.017957444115883448;
                  } else {
                    result[0] += 0.03904569936255059;
                  }
                } else {
                  result[0] += -0.09564638805872659;
                }
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.724856853485109198) ) ) {
                result[0] += -0.002098456352661508;
              } else {
                result[0] += -0.03011112778781884;
              }
            }
          }
        } else {
          result[0] += -0.031215603317088603;
        }
      }
    }
  }
  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.91160488128662287) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)63.50000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += 0.017351463758868767;
              } else {
                result[0] += -0.030316342271121816;
              }
            } else {
              result[0] += -0.003762047185894303;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += -0.0152748252149192;
            } else {
              result[0] += 0.003524192620565254;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
            result[0] += 0.008604183260833375;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.504379272460938388) ) ) {
              result[0] += 0.001824478578186862;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.03411140029279432;
              } else {
                result[0] += 0.0006608085303202604;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.03698199200667716;
              } else {
                result[0] += 0.010262110195680534;
              }
            } else {
              result[0] += -0.02879643238820921;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.131699204444885698) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37995386123657404) ) ) {
                  result[0] += -0.06388004456964934;
                } else {
                  result[0] += 0.06573583664085726;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.02434244575367185;
                } else {
                  result[0] += -0.073340239124115;
                }
              }
            } else {
              result[0] += 0.009115075642322957;
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += -0.011618057417287851;
              } else {
                result[0] += -0.04418461449001881;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.0022138409787549147;
              } else {
                result[0] += -0.04551409346111463;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.08025836664957624;
                } else {
                  result[0] += -0.04325343804559419;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)31.50000000000000355) ) ) {
                  result[0] += 0.043739474149157936;
                } else {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)199.5000000000000284) ) ) {
                      result[0] += 0.019114004038691776;
                    } else {
                      result[0] += -0.0051422759854791835;
                    }
                  } else {
                    result[0] += -0.02347424484047146;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.05404037830931257;
                } else {
                  result[0] += -0.0038009248272665706;
                }
              } else {
                result[0] += 0.011326307370278541;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.681859493255617011) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)34.50000000000000711) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.285887241363526279) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)19.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0008118348504998039;
                  } else {
                    result[0] += -0.04458603776711747;
                  }
                } else {
                  result[0] += -0.04554784254908767;
                }
              } else {
                result[0] += 0.007136472064581141;
              }
            } else {
              result[0] += -0.08337020109210058;
            }
          } else {
            result[0] += 0.000892002885484642;
          }
        } else {
          result[0] += 0.002517111241530773;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                result[0] += 0.0021975414187578933;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0292482039340994;
                    } else {
                      result[0] += 0.0064327475436788024;
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)35.50000000000000711) ) ) {
                      result[0] += -1.3994727112310847e-05;
                    } else {
                      result[0] += -0.06313252012091647;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78735828399658381) ) ) {
                    result[0] += -0.00030245481664940187;
                  } else {
                    result[0] += 0.03467102358260022;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.79835033416748225) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.08233521153124723;
                    } else {
                      result[0] += -0.0005295504182541362;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.07933227794757156;
                    } else {
                      result[0] += -0.015522179740151916;
                    }
                  }
                } else {
                  result[0] += -0.05359694386932661;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.015141099743683169;
                  } else {
                    result[0] += -0.012986038099684835;
                  }
                } else {
                  result[0] += -0.01198507517258424;
                }
              }
            }
          } else {
            result[0] += -0.015131686335542;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
            result[0] += -0.011449773664084145;
          } else {
            result[0] += -0.04309664784212883;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)111.5000000000000142) ) ) {
        result[0] += -0.025493572492597472;
      } else {
        result[0] += 0.0014383289348429192;
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.01647395239083478;
            } else {
              result[0] += -0.04954207457339519;
            }
          } else {
            result[0] += -0.003258694308196064;
          }
        } else {
          result[0] += -0.05275879340434672;
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)164.5000000000000284) ) ) {
          result[0] += 0.06482753180744001;
        } else {
          result[0] += -0.0043398168921210045;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)47.50000000000000711) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.020949519226385837;
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.166635274887085849) ) ) {
            result[0] += 0.06066564263510441;
          } else {
            result[0] += -0.03245638347939201;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += 0.09012471416154279;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
              result[0] += 0.01037348185916558;
            } else {
              result[0] += -0.08426896231253617;
            }
          } else {
            result[0] += 0.02867763611154622;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.10377502441406428) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
              result[0] += -0.1425106475535156;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.010479968430816841;
              } else {
                result[0] += 0.04969367813315051;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)67.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.868834793567657693) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += -0.09940225082604065;
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
                      result[0] += 0.09389331405152399;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.793003082275392401) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                          result[0] += -0.00010919271514514442;
                        } else {
                          result[0] += -0.10870752875072126;
                        }
                      } else {
                        result[0] += 0.01765406062893028;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    result[0] += 0.031375114580254826;
                  } else {
                    result[0] += -0.007319613911092836;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.154959201812744585) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += -0.023433667992266344;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.11326837539672896) ) ) {
                      result[0] += 0.13278207703605585;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.661214828491211826) ) ) {
                        result[0] += -0.05097567594036248;
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.736373662948609287) ) ) {
                          result[0] += 0.14514336771197436;
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.540854334831238237) ) ) {
                            result[0] += -0.0713991669143175;
                          } else {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.942183732986451083) ) ) {
                              result[0] += 0.09877597313474838;
                            } else {
                              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.48298668861389249) ) ) {
                                result[0] += -0.023675808564834647;
                              } else {
                                result[0] += 0.05674776724088507;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += -0.03895986692030472;
                    } else {
                      result[0] += 0.01479829606821857;
                    }
                  } else {
                    result[0] += -0.022160033230553683;
                  }
                }
              }
            } else {
              result[0] += -0.057786197008066024;
            }
          }
        } else {
          result[0] += -0.02050021820073606;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.03824654547833134;
        } else {
          result[0] += -0.009333844033746892;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        result[0] += 0.0006356856407820061;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            result[0] += -0.007850911609208838;
          } else {
            result[0] += -0.08603916194400325;
          }
        } else {
          result[0] += 0.06952687042475086;
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 0.03562465364301627;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.03409783001165025;
            } else {
              result[0] += 0.06393568094584005;
            }
          }
        } else {
          result[0] += -0.00378619906220974;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.03441427320732859;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                  result[0] += 0.008883926896192579;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.206374883651734287) ) ) {
                    result[0] += 0.006522783960881054;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.022297825456667225;
                        } else {
                          result[0] += -0.06319157186526232;
                        }
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.024549846755674197;
                          } else {
                            result[0] += 0.10271424238272985;
                          }
                        } else {
                          result[0] += -0.021572137867170136;
                        }
                      }
                    } else {
                      result[0] += 0.0035396426237448045;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
                result[0] += -0.028285125754579872;
              } else {
                result[0] += -0.07585985638514557;
              }
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.019234911416622813;
              } else {
                result[0] += -0.0587676971481933;
              }
            } else {
              result[0] += -0.09593826202653857;
            }
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                result[0] += -0.06214670541084341;
              } else {
                result[0] += 0.0017535107906899224;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += 0.021691968496362127;
                } else {
                  result[0] += -0.0027554769998014393;
                }
              } else {
                result[0] += 0.05110224039748873;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.725620865821838823) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.004138026726501487;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.280697107315064365) ) ) {
                  result[0] += -0.0002610945584878428;
                } else {
                  result[0] += -0.06091604105244496;
                }
              }
            } else {
              result[0] += -0.047608152205328796;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
            result[0] += 0.0011820951398164668;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)147.5000000000000284) ) ) {
              result[0] += -0.04381788086050515;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                result[0] += -0.0663457696073564;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                    result[0] += -0.09058961565148735;
                  } else {
                    result[0] += 0.04785991636826978;
                  }
                } else {
                  result[0] += 0.034107650287707895;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.008082101463801698;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.045479452710899246;
              } else {
                result[0] += -0.008513474594376309;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)185.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.08073929474553382;
                } else {
                  result[0] += -0.03240800028087387;
                }
              } else {
                result[0] += -0.015201013909318906;
              }
            }
          }
        }
      } else {
        result[0] += 0.003123199794709405;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)219.5000000000000284) ) ) {
            result[0] += 0.010496835818988031;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.05398836278834265;
              } else {
                result[0] += -0.006027597281613305;
              }
            } else {
              result[0] += -0.02028141574094107;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
              result[0] += -0.010428873125875357;
            } else {
              result[0] += -0.05436403910318921;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
              result[0] += 0.010352981498130392;
            } else {
              result[0] += -0.02300092541719112;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.0852242533226388;
            } else {
              result[0] += -0.03221687981566243;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
              result[0] += -0.011406867443922367;
            } else {
              result[0] += 0.02548321858296755;
            }
          }
        } else {
          result[0] += -0.036680930281308;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0028125135780061633;
          } else {
            result[0] += 0.029792365822789185;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += -0.018657002472567852;
              } else {
                result[0] += 0.04079577990822887;
              }
            } else {
              result[0] += 0.013867868382863514;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
                result[0] += -0.028454879856954954;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.214365959167481357) ) ) {
                  result[0] += 0.007937273718519992;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)207.5000000000000284) ) ) {
                    result[0] += -0.03866208823071674;
                  } else {
                    result[0] += 0.009817219770586135;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.280697107315064365) ) ) {
                  result[0] += -0.010582625314790949;
                } else {
                  result[0] += -0.0382181469102704;
                }
              } else {
                result[0] += 0.024927841571228855;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.04170075403133968;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.02093537976888354;
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += 0.01956514446141769;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                    result[0] += -0.01096373543786902;
                  } else {
                    result[0] += -0.03912309909145797;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
                    result[0] += 0.002937662178573639;
                  } else {
                    result[0] += -0.0592961495164932;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.002514272259014414;
              } else {
                result[0] += -0.015643319820221054;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += 0.02390492123957744;
              } else {
                result[0] += 0.0024712031376945102;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                result[0] += -0.029788880996808282;
              } else {
                result[0] += 0.0589235996898718;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)309.5000000000000568) ) ) {
                  result[0] += -0.01447270311950359;
                } else {
                  result[0] += 0.029087814211313054;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.050441672125498827;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.777633190155030185) ) ) {
                    result[0] += -0.035277418753420685;
                  } else {
                    result[0] += 0.027679714709681946;
                  }
                }
              }
            } else {
              result[0] += -0.00041908904560007817;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
            result[0] += -0.0022481393449978945;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.027040985316097988;
            } else {
              result[0] += -0.009749475123997358;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
            result[0] += -0.023782292773403366;
          } else {
            result[0] += -0.05720553914600014;
          }
        } else {
          result[0] += 0.022324291286540083;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
        result[0] += -0.00032611743845497004;
      } else {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += 0.016035554176581958;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.04056891216995163;
              } else {
                result[0] += -0.004820189940055543;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.5513958930969256) ) ) {
                  result[0] += -0.0008912172485315904;
                } else {
                  result[0] += 0.01876869898317378;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.030313403437928533;
                  } else {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.011018147838793915;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.87636661529541193) ) ) {
                        result[0] += 0.0059120465218861955;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.008057937175084275;
                        } else {
                          result[0] += -0.0469343875363211;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.034331798747306456;
                }
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.010689526209298847;
                } else {
                  result[0] += 0.009309815362130913;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += -0.010394711608061368;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.89450073242187678) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += 0.013865517500144364;
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.03595115223286325;
                        } else {
                          result[0] += -0.0030779661047609422;
                        }
                      }
                    } else {
                      result[0] += 0.04864610873385461;
                    }
                  } else {
                    result[0] += 0.017407202368136743;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.01911377783684664;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.03203731781296377;
        } else {
          result[0] += -0.030348652768504892;
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
              result[0] += -0.054663202783694145;
            } else {
              result[0] += 0.018377156303958347;
            }
          } else {
            result[0] += -0.04485480768248406;
          }
        } else {
          if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.013506646573078367;
            } else {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.04087588554742914;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03632588729885335;
                } else {
                  result[0] += -0.03435623232057038;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.921100616455079013) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.0004250867592046898;
                } else {
                  result[0] += -0.05575147633301656;
                }
              } else {
                result[0] += 0.021814372907096397;
              }
            } else {
              result[0] += -0.05715978685738204;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      result[0] += 8.150148787870368e-05;
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.016531057764097055;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
                    result[0] += -0.05117020543408814;
                  } else {
                    result[0] += -0.0049441638814416156;
                  }
                } else {
                  result[0] += 0.039706415457711954;
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.02299747217022257;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.01767504852859649;
                  } else {
                    result[0] += -0.07414235239793764;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.78207492828369318) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.084203958511353427) ) ) {
                      result[0] += -0.025669313843109623;
                    } else {
                      result[0] += 0.03849081507120425;
                    }
                  } else {
                    result[0] += 0.045899039612131785;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
              result[0] += -0.013919681719731472;
            } else {
              result[0] += 0.006135473077386377;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
            result[0] += -0.003838011715155706;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.09893783086169512;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += -0.03543682889323699;
                  } else {
                    result[0] += -0.11228466683487946;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.53498554229736506) ) ) {
                  result[0] += 0.012472306276758549;
                } else {
                  result[0] += -0.029116158322928538;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.70078086853027521) ) ) {
                result[0] += -0.01478399452266213;
              } else {
                result[0] += 0.022785539312303732;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.04049275803604053;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.005962535070648672;
              } else {
                result[0] += 0.05905773070658135;
              }
            } else {
              result[0] += -0.06301030708349492;
            }
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.69067406654357999) ) ) {
                result[0] += 0.019248065361737326;
              } else {
                result[0] += -0.07830081700184573;
              }
            } else {
              result[0] += 0.008050871836252043;
            }
          } else {
            result[0] += -0.03520677770351639;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY(  (data[35].missing != -1) && (data[35].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.10884105309687514;
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
        result[0] += -0.0015713921836885067;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.003923622446453344;
            } else {
              result[0] += -0.08582674113832203;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.914472818374634233) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.37576770782470881) ) ) {
                result[0] += 0.005474832733476378;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.08774855711434516;
                } else {
                  result[0] += -0.0007244794230444086;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.04854813282019143;
                } else {
                  result[0] += -0.06961089223772382;
                }
              } else {
                result[0] += -0.014086602516944142;
              }
            }
          }
        } else {
          result[0] += -0.01131528341170785;
        }
      }
    }
  } else {
    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)227.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
              result[0] += -0.0018196379850207306;
            } else {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.439558982849121982) ) ) {
                  result[0] += -0.040534000589106875;
                } else {
                  if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.455636501312257636) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                        result[0] += -0.04485478238153913;
                      } else {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                          result[0] += -0.09148180446100844;
                        } else {
                          result[0] += -0.005003104081693393;
                        }
                      }
                    } else {
                      result[0] += 0.030390108897041874;
                    }
                  } else {
                    result[0] += -0.0008136078120329944;
                  }
                }
              } else {
                result[0] += -0.046836621291451136;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.089241743087769443) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += -0.04639361234497222;
                } else {
                  result[0] += -0.005644154643992385;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.03995243845835182;
                  } else {
                    result[0] += 0.012131381408067847;
                  }
                } else {
                  if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.03178202719762667;
                  } else {
                    result[0] += 0.013547706171780942;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.028327001087125617;
              } else {
                result[0] += 0.003138822975731644;
              }
            }
          }
        } else {
          result[0] += 0.005244240540437375;
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.01565877373382584;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.00805138065532682;
              } else {
                result[0] += 0.01939937375768678;
              }
            }
          } else {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.011350512203374626;
            } else {
              result[0] += 0.03035842870901011;
            }
          }
        } else {
          result[0] += -0.007651797898839594;
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.017188672778305592;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                result[0] += 0.0005926275897739667;
              } else {
                result[0] += -0.022292248570892115;
              }
            }
          } else {
            result[0] += 0.011819254798488393;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)80.50000000000001421) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.001351356506349433) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.014751434083101592;
                  } else {
                    result[0] += 0.01217686493132102;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.014702521320295262;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                      result[0] += -0.008722934895082527;
                    } else {
                      result[0] += -0.0571820497987072;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.02728242176586786;
                  } else {
                    result[0] += -0.004383812113022129;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.00441208895485683;
                    } else {
                      result[0] += -0.026113086410258935;
                    }
                  } else {
                    result[0] += 0.016920957514366227;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.009872545099325733;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.07104238919448717;
                  } else {
                    result[0] += -0.0015331628232955331;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += 0.012167956249900769;
                } else {
                  result[0] += -0.0276670394022018;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.004895895029694505;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                result[0] += 0.004636982614807268;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.037794637904282606;
                } else {
                  result[0] += -0.02223885364922659;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.506659984588624823) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.051912069320679599) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.03531382456823364;
              } else {
                result[0] += -0.01225195542242869;
              }
            } else {
              result[0] += -0.0007200826897408349;
            }
          } else {
            result[0] += 0.0044776063949893;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.014180913935859435;
          } else {
            result[0] += -0.03438751125512698;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.363266706466675693) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += 3.6711081186333665e-05;
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
            result[0] += -0.05032271647351261;
          } else {
            result[0] += -0.009694918523249411;
          }
        }
      } else {
        result[0] += 0.0029283954869303874;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.028861761093140537) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
              result[0] += -0.0027633043417703528;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += -0.007252736936791309;
              } else {
                result[0] += -0.05372561376455797;
              }
            }
          } else {
            result[0] += 0.009740327416888177;
          }
        } else {
          result[0] += 0.007027002298230187;
        }
      } else {
        result[0] += -0.018496252437658595;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)207.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.014776448172230075;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
                  result[0] += 0.07373312903759939;
                } else {
                  result[0] += -0.018090377186077963;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.851041555404663974) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0868348614406887;
                  } else {
                    result[0] += -0.02770498915159375;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.649621725082398349) ) ) {
                    result[0] += -0.034074868991604175;
                  } else {
                    result[0] += 0.014291209035985106;
                  }
                }
              } else {
                result[0] += -0.09033597461066073;
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                result[0] += 0.017290529549662325;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)288.5000000000000568) ) ) {
                  result[0] += -0.03172467075010097;
                } else {
                  result[0] += 0.0067852489082366844;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.03233952020897305;
                } else {
                  result[0] += 0.06548101787989703;
                }
              } else {
                result[0] += 0.041896662925011374;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.006567910870931714;
          } else {
            result[0] += -0.04770009400248723;
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.02622828015819528;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
            result[0] += 0.002210807065538344;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += -0.007276279855102443;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
                result[0] += -0.01510999401974944;
              } else {
                result[0] += -0.10190592434492858;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.28360033035278498) ) ) {
                  result[0] += 0.004059782612981336;
                } else {
                  result[0] += 0.08437158743221967;
                }
              } else {
                result[0] += 0.002868643242208512;
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.061421104271771144;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                  result[0] += -0.015871484132276178;
                } else {
                  result[0] += 0.07377430026111446;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.826510190963745561) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.035906891654129656;
                } else {
                  result[0] += 0.006007810923272151;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0604278875858572;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)282.5000000000000568) ) ) {
                      result[0] += -0.038740829832840586;
                    } else {
                      result[0] += 0.003442132538357598;
                    }
                  } else {
                    result[0] += -0.0047354220297920685;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.821564435958863193) ) ) {
                result[0] += 0.0022815332081160974;
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.0014424436670528026;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.08525625584710794;
                    } else {
                      result[0] += 0.05268250934946204;
                    }
                  } else {
                    result[0] += -0.12509427978970902;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.549732685089113104) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                result[0] += 0.017638135093517665;
              } else {
                result[0] += -0.012304052269987376;
              }
            } else {
              result[0] += 0.02225555159028968;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)167.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.03201255751685724;
                  } else {
                    result[0] += 0.00318239091080901;
                  }
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                        result[0] += 0.06167853325790843;
                      } else {
                        result[0] += -0.07596843109068319;
                      }
                    } else {
                      result[0] += 0.017951288014317415;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.0501932498034574;
                    } else {
                      result[0] += -0.006676345477767471;
                    }
                  }
                }
              } else {
                result[0] += 0.01941207944260224;
              }
            } else {
              result[0] += -0.02495194905015725;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
            result[0] += -0.02416382653898524;
          } else {
            result[0] += -0.054337258762333786;
          }
        } else {
          result[0] += 0.02283211311500511;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.10863716016335223;
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
        result[0] += -0.0016522217066408285;
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.610145330429078037) ) ) {
                result[0] += 0.0021101907670715006;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.04690878690970334;
                } else {
                  result[0] += -0.06897768981683951;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0007141324866415473;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.057316518499238446;
                  } else {
                    result[0] += 0.0019612399884494204;
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.050418739467507204;
                } else {
                  result[0] += -0.11260868254204859;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.41589117050171076) ) ) {
                result[0] += -0.020093783533763912;
              } else {
                result[0] += -0.06508622932259049;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.026309401223868347;
              } else {
                result[0] += -0.015053052214787397;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += -0.08129340584753689;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.05673785904988735;
                } else {
                  result[0] += -0.017068520131059747;
                }
              } else {
                result[0] += -0.008042345339771429;
              }
            }
          } else {
            result[0] += -0.005089602474621133;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
        result[0] += 0.00015647906761979556;
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.44809746742248624) ) ) {
                  result[0] += 0.012950114565276097;
                } else {
                  result[0] += 0.05316517529102698;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.350240230560303178) ) ) {
                  result[0] += 0.0003579837145174968;
                } else {
                  result[0] += -0.08449532318070058;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.03846945467572551;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0026583550729558677;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.04257496769945665;
                    } else {
                      result[0] += -0.010637711697710472;
                    }
                  }
                }
              } else {
                result[0] += 0.026739534079082646;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.510617971420288974) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += -0.002983526486526903;
                } else {
                  result[0] += 0.011588601176253738;
                }
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.007380385233973033;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07093748265187882;
                  } else {
                    result[0] += -0.02659072698208071;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
                      result[0] += 0.04791922850921139;
                    } else {
                      result[0] += 0.008665900082330025;
                    }
                  } else {
                    result[0] += -0.0032382548890905897;
                  }
                } else {
                  result[0] += -0.029802604967757246;
                }
              } else {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.00462876450782028;
                  } else {
                    result[0] += 0.019956352400874473;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.05429658692903405;
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.921100616455079013) ) ) {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.0016299851889965554;
                            } else {
                              result[0] += -0.036700754868122326;
                            }
                          } else {
                            result[0] += 0.01130155442722075;
                          }
                        } else {
                          result[0] += 0.02193880500755258;
                        }
                      } else {
                        result[0] += 0.017597464444615518;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += -0.016096671625804946;
                    } else {
                      result[0] += 0.027709697550618324;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            result[0] += -0.03144223456546164;
          } else {
            result[0] += 0.0286877659226578;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.03436140498956316;
            } else {
              result[0] += 0.040773391764408425;
            }
          } else {
            result[0] += -0.02889589990687242;
          }
        } else {
          if ( LIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
              result[0] += 0.012183600407569405;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                  result[0] += 0.0014408040828349242;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += 0.03856615091893544;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.04955502307900643;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.003174711682876607;
                      } else {
                        result[0] += -0.04143730499392609;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.023271305981343932;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              result[0] += -0.025442789570575487;
            } else {
              result[0] += 0.008310344516689886;
            }
          }
        }
      } else {
        result[0] += -0.028368544711874416;
      }
    }
  }
  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
    result[0] += -0.0011891208775480263;
  } else {
    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)87.50000000000001421) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.11973339436486707;
          } else {
            result[0] += -0.006833301606210232;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)190.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                      result[0] += -0.02755541069965873;
                    } else {
                      result[0] += 0.03211593041807779;
                    }
                  } else {
                    result[0] += -0.04485999294030022;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.07153280075207685;
                  } else {
                    result[0] += -0.005219092752277299;
                  }
                }
              } else {
                result[0] += 0.02025787202656173;
              }
            } else {
              result[0] += 0.05541206147081053;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)278.5000000000000568) ) ) {
              result[0] += -0.005565327725151474;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)299.5000000000000568) ) ) {
                  result[0] += -0.008144858308506739;
                } else {
                  result[0] += 0.06996946066909655;
                }
              } else {
                if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                        result[0] += 0.07924278438785301;
                      } else {
                        result[0] += 0.00833186462062399;
                      }
                    } else {
                      result[0] += -0.06765120355998218;
                    }
                  } else {
                    result[0] += 0.11333519283319192;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.04388746402402775;
                  } else {
                    result[0] += 0.0081419921853093;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += 0.019281011050559033;
        } else {
          result[0] += -0.020826130264693515;
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)69.50000000000001421) ) ) {
                    result[0] += -0.0987471359497259;
                  } else {
                    result[0] += -0.007410881685206976;
                  }
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.019382082561509566;
                  } else {
                    result[0] += 0.009761700425316634;
                  }
                }
              } else {
                result[0] += 0.0220826093081398;
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += 0.019154351265073616;
                  } else {
                    result[0] += 0.06686073731245538;
                  }
                } else {
                  result[0] += 0.0008799330754580512;
                }
              } else {
                result[0] += 0.0021167848453137136;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += -0.11771809651526229;
              } else {
                result[0] += -0.01513952126129606;
              }
            } else {
              result[0] += 0.029620495370257238;
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.005603675222852539;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.0022106404525402373;
                } else {
                  result[0] += -0.028594429538927008;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.285166740417482245) ) ) {
                  result[0] += -0.021434704261105778;
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += -0.0495816783940024;
                  } else {
                    if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04850381762958966;
                    } else {
                      result[0] += 0.010768876892585173;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.06204470575671082;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.892608642578125888) ) ) {
                  result[0] += 0.0561749743680008;
                } else {
                  result[0] += -0.03210783637311337;
                }
              } else {
                result[0] += -0.04043123695535281;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.445957899093628818) ) ) {
                  result[0] += 0.017494591634536868;
                } else {
                  result[0] += 0.13455097712822836;
                }
              } else {
                result[0] += -0.02389010518480246;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += 0.00492778434507612;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.43260431289673029) ) ) {
                result[0] += -0.03326234797292203;
              } else {
                result[0] += -0.08087548989743865;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)294.5000000000000568) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += -0.0029678768855258865;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.00022495837820931988;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    result[0] += 0.03442045426562192;
                  } else {
                    result[0] += -0.03814413640606246;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                    result[0] += 0.0026909089925347103;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.06116221151194878;
                    } else {
                      result[0] += 0.04194355344942319;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.576162815093995917) ) ) {
                      result[0] += 0.01034246665229654;
                    } else {
                      result[0] += -0.03232166018161035;
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.019498988873439746;
                    } else {
                      result[0] += 0.0576051283980529;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
              result[0] += 0.001434703130345564;
            } else {
              result[0] += -0.014719539394581656;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.795762062072754794) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
            result[0] += -0.010351871496390615;
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.008164019488231207;
            } else {
              result[0] += -0.0850915415621063;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.973273515701294833) ) ) {
              result[0] += -0.02838087230353306;
            } else {
              result[0] += 0.0032666968419347584;
            }
          } else {
            result[0] += 0.01568831395878428;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
            result[0] += 0.015914212208621766;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.367881059646607333) ) ) {
              result[0] += -0.051770141598301046;
            } else {
              result[0] += 0.022707732461427866;
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)34.50000000000000711) ) ) {
            result[0] += 0.00940980761534134;
          } else {
            result[0] += 0.041737101747911294;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.363078355789185458) ) ) {
          result[0] += -0.0016534842409162969;
        } else {
          result[0] += -0.007236734348438959;
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
              result[0] += -0.08229647306239783;
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)46.50000000000000711) ) ) {
                result[0] += -0.00084646238015313;
              } else {
                result[0] += -0.04745136081842522;
              }
            }
          } else {
            result[0] += -0.06900999378660505;
          }
        } else {
          result[0] += 0.0932847922508569;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
        result[0] += -0.04419868598960682;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.05807435924824272;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
            result[0] += 0.10851987982060692;
          } else {
            result[0] += 0.004581675153406958;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.020127415657043901) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37995386123657404) ) ) {
            result[0] += -0.059650188044913566;
          } else {
            result[0] += 0.06122730250929778;
          }
        } else {
          if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.826510190963745561) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.010308530329155088;
              } else {
                result[0] += 0.02719458710872329;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.008756485450876208;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.47345590591430842) ) ) {
                      result[0] += 0.10761675369181578;
                    } else {
                      result[0] += 0.042702565954876794;
                    }
                  }
                } else {
                  result[0] += -0.01118346451254649;
                }
              } else {
                result[0] += 0.007114480351672986;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)296.5000000000000568) ) ) {
                  result[0] += 0.020263732804938206;
                } else {
                  result[0] += 0.1527255914807482;
                }
              } else {
                result[0] += -0.026252081107505057;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.010340564319950916;
              } else {
                result[0] += -0.07874514100071117;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.03230552199748701;
              } else {
                result[0] += -0.009263515735628497;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.03708311343960595;
                    } else {
                      result[0] += 0.003715612070242438;
                    }
                  } else {
                    result[0] += -0.029702758659874286;
                  }
                } else {
                  result[0] += -0.03356710265967434;
                }
              } else {
                result[0] += -0.049951357806791906;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
              result[0] += -0.03287642609387914;
            } else {
              result[0] += -0.07453963500289655;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
            result[0] += 0.00010059426943549307;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                  result[0] += -0.002608047183299179;
                } else {
                  result[0] += -0.03978289413994821;
                }
              } else {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.0925779342651385) ) ) {
                          result[0] += -0.005996819107004881;
                        } else {
                          result[0] += 0.01888966432947452;
                        }
                      } else {
                        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                          result[0] += -0.021536597207767808;
                        } else {
                          result[0] += 0.0048394865457172705;
                        }
                      }
                    } else {
                      result[0] += -0.04520872842146888;
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)192.5000000000000284) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)59.50000000000000711) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.47345590591430842) ) ) {
                          result[0] += 0.0009002114671592467;
                        } else {
                          result[0] += 0.027600543550809827;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                          result[0] += 0.018904501958825787;
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.020265472501928224;
                          } else {
                            result[0] += 0.029748943583952664;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[58].missing != -1) || (data[58].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.07063868727236931;
                        } else {
                          result[0] += -0.014677000873528337;
                        }
                      } else {
                        result[0] += 0.00369990809568076;
                      }
                    }
                  }
                } else {
                  result[0] += -0.020872125304692908;
                }
              }
            } else {
              result[0] += -0.025226173854003237;
            }
          }
        }
      }
    }
  }
}

