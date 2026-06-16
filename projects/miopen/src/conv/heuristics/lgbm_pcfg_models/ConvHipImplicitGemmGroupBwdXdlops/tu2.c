
#include "header.h"

void predict_unit2(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0014671227579852106;
            } else {
              result[0] += -0.04148981195276069;
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.0030424337580636156;
            } else {
              result[0] += 0.027108560458034305;
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.0724329560855084;
          } else {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.007693746009437057;
            } else {
              result[0] += 0.005046943426690444;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
            result[0] += -0.0004967124885355268;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.032886059757002976;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0006609017334822409;
              } else {
                result[0] += 0.09766159706749931;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
            result[0] += -0.027033152769765745;
          } else {
            result[0] += -0.0974379028531614;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.0006659894248790655;
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
          result[0] += 0.002258602674809906;
        } else {
          result[0] += -0.0349817481364204;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.059557789192515725;
            } else {
              result[0] += -0.006361152442814883;
            }
          } else {
            result[0] += -0.04473135977544523;
          }
        } else {
          result[0] += -0.039103539735253996;
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.02604460716247603) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.011918380333693877;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.006368197265231934;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                        result[0] += -0.0019233709939724575;
                      } else {
                        result[0] += 0.036590986198889015;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
                      result[0] += -0.056008396904836694;
                    } else {
                      result[0] += -0.004384610403466726;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.030161617213898526;
                } else {
                  result[0] += -0.05077808330435932;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.04456535409070776;
                  } else {
                    result[0] += 0.01540825180950365;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += 0.020712002704192087;
                  } else {
                    result[0] += 0.06521320612842249;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                  result[0] += -0.013253420341505048;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                        result[0] += 0.0023445931286943944;
                      } else {
                        result[0] += -0.06340573713632813;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += 0.003993230727411662;
                      } else {
                        result[0] += 0.03160274183019679;
                      }
                    }
                  } else {
                    result[0] += -0.009848239725776742;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += 0.051583563363550036;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
                    result[0] += 0.007860102475800787;
                  } else {
                    result[0] += -0.03866317017733912;
                  }
                }
              } else {
                result[0] += -0.022625530497636433;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.00011446080922649159;
                    } else {
                      result[0] += 0.057738699117986486;
                    }
                  } else {
                    result[0] += 0.04426374678110063;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.01004036064010082;
                  } else {
                    result[0] += -0.04781340692084636;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                  result[0] += 0.004689165017416006;
                } else {
                  result[0] += 0.1043984139791968;
                }
              }
            }
          }
        } else {
          result[0] += -0.032741704771859365;
        }
      }
    } else {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.005233585611918817;
            } else {
              result[0] += -0.05485003899783158;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.00956091307916116;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.0591679175542352;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += -0.05696108257058973;
                } else {
                  result[0] += 0.019760364006766284;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.08911709959591989;
              } else {
                result[0] += -0.011529254217256596;
              }
            } else {
              result[0] += 0.02776575823835091;
            }
          } else {
            result[0] += -0.045326970361465;
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.011229482277973433;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.026078176674283766;
          } else {
            result[0] += -0.061429071116119394;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += 0.010864954603730735;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.0020214865094341965;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.029435383938105443;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
                    result[0] += -0.11719402849571109;
                  } else {
                    result[0] += -0.018963627875050865;
                  }
                }
              }
            } else {
              result[0] += 0.0031043631151101867;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.535966873168947089) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.00028148019394653;
            } else {
              result[0] += -0.0398049508908499;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.04710933083481221;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.91117334365844904) ) ) {
                      result[0] += -0.0277044056367662;
                    } else {
                      result[0] += 0.013469876285493393;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0106596624988618;
                  } else {
                    result[0] += 0.029282594182969413;
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += 0.012874606957912422;
                  } else {
                    result[0] += -0.014225106607533398;
                  }
                } else {
                  result[0] += -0.037597244793869444;
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += -0.002727767772052133;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.02078034166222936;
                    } else {
                      result[0] += -0.07193548948960186;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.001381844877065171;
                    } else {
                      result[0] += 0.07673047001112024;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.007886142120486305;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.011636585577138028;
                      } else {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.06248836607491581;
                        } else {
                          result[0] += 0.042965399262520354;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0558624195255364;
                  } else {
                    result[0] += -0.01789461182546872;
                  }
                } else {
                  result[0] += -0.01871533982292757;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.016916999927980302;
          } else {
            result[0] += -0.004007963104415428;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
            result[0] += 0.02193295746706074;
          } else {
            result[0] += -0.010534743331370866;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.0024677239613740687;
      } else {
        result[0] += 0.0028631841600940903;
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += -0.004150663332390061;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
                  result[0] += 0.009435834347260705;
                } else {
                  result[0] += -0.026645366476558394;
                }
              } else {
                result[0] += 0.016116213253315546;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                result[0] += 0.0370207893849034;
              } else {
                result[0] += -0.023046765344892257;
              }
            }
          }
        } else {
          result[0] += -0.038838561660094445;
        }
      } else {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.005293686467637927;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
              result[0] += 0.00884992932230103;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.048771738806703234;
                } else {
                  result[0] += -0.0005203322547326256;
                }
              } else {
                result[0] += -0.0730392085014722;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
              result[0] += 0.0061023304909274614;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.07560477197251952;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                    result[0] += -0.058240152305680574;
                  } else {
                    result[0] += 0.05789290625191193;
                  }
                } else {
                  result[0] += 0.10434371507980478;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.0017147480718185323;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.433652400970459873) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
            result[0] += -0.004167519192768038;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
              result[0] += -0.07176973764195614;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += 0.06179301508426867;
                  } else {
                    result[0] += 0.013360821203374493;
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                        result[0] += -0.11054181500699806;
                      } else {
                        result[0] += -0.019299809194355408;
                      }
                    } else {
                      result[0] += 0.028795514273441892;
                    }
                  } else {
                    result[0] += 0.028423923157551042;
                  }
                }
              } else {
                result[0] += 0.002333559171690441;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
            result[0] += -0.006439980596586074;
          } else {
            result[0] += -0.02596236627749919;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.005874327039979945;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.014515175202081915;
                } else {
                  result[0] += -0.0015314304001755367;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.509355545043946201) ) ) {
                    result[0] += -0.008277506808536571;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.022112867496098667;
                    } else {
                      result[0] += -0.07588582653698861;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.02086310485432494;
                    } else {
                      result[0] += -0.001949581400496076;
                    }
                  } else {
                    result[0] += 0.008496329163365006;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += 0.002312484937750233;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.04419094561326714;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += -0.02332357503911919;
                  } else {
                    result[0] += 0.0029595636966988762;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += -0.014790272873699127;
                  } else {
                    result[0] += 0.011037087291573273;
                  }
                } else {
                  result[0] += -0.0217771140263049;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
                  result[0] += 0.009523146511504911;
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += 0.07882363326325068;
                        } else {
                          result[0] += 0.005009687199941074;
                        }
                      } else {
                        result[0] += -0.019018216042706933;
                      }
                    } else {
                      result[0] += 0.03531019717700741;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.0010350143450352847;
                        } else {
                          result[0] += 0.050007007658442564;
                        }
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.022373113259760755;
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.0207538642310016;
                          } else {
                            result[0] += -0.11455275253737306;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += 0.002617817491009565;
                      } else {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                            result[0] += 0.02986521061420382;
                          } else {
                            result[0] += -0.06493302221818946;
                          }
                        } else {
                          result[0] += 0.053122603146692704;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.00507047676149829;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.559112548828125888) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.018487997145528255;
                } else {
                  result[0] += -0.06889101825016336;
                }
              } else {
                result[0] += -0.002905965284662416;
              }
            }
          } else {
            result[0] += 0.025309647231748795;
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0008930063773808692;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += -0.0014542453678636422;
              } else {
                result[0] += -0.03821191184277131;
              }
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.049435998338612244;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += -0.04791443799568713;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.03290465461993503;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.03087808653325651;
                  } else {
                    result[0] += -0.01244253526082767;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.004275862304117206;
            } else {
              result[0] += 0.04381273576846606;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.921924352645874468) ) ) {
              result[0] += -0.06808481638717571;
            } else {
              result[0] += 0.013873286904068761;
            }
          }
        }
      }
    } else {
      result[0] += 0.007055412534487011;
    }
  } else {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += 0.015000039202430496;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.197173833847046787) ) ) {
                result[0] += -0.0022847614078365;
              } else {
                result[0] += -0.019704055934338667;
              }
            }
          } else {
            result[0] += 0.0030170237509922083;
          }
        } else {
          result[0] += -0.01137498092757857;
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0024955370299857;
          } else {
            result[0] += -0.019231138446532042;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.0319837028585677;
          } else {
            result[0] += 0.04345083717997511;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.0029754524282337943;
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.012576733492433415;
          } else {
            result[0] += -0.04412983367396025;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
              result[0] += -0.027054406493582955;
            } else {
              result[0] += 0.011752359491746876;
            }
          } else {
            result[0] += -0.04695340247827242;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
        result[0] += -0.001038709615651341;
      } else {
        if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.1095643446886897;
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
                result[0] += -0.0009921610507377142;
              } else {
                result[0] += -0.05644939360554562;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.04073610907581637;
              } else {
                result[0] += -0.008041750593271486;
              }
            }
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.011256013277159964;
              } else {
                result[0] += -0.1450519770917958;
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.011994386184214438;
              } else {
                result[0] += 0.04033953306488455;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.028861761093140537) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
            result[0] += -0.03103857038605364;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
              result[0] += -0.009830806583344615;
            } else {
              result[0] += 0.007826403394362571;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.040285587310792792) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                    result[0] += 0.0314581961234083;
                  } else {
                    result[0] += -0.034139154295040074;
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.695914268493653232) ) ) {
                    result[0] += -0.07578011857418489;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                        result[0] += 0.10458120195248187;
                      } else {
                        result[0] += -0.07789766412312177;
                      }
                    } else {
                      result[0] += -0.04666876534883074;
                    }
                  }
                }
              } else {
                result[0] += -0.07268950846023774;
              }
            } else {
              result[0] += 0.1550798925351955;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                result[0] += 0.00479833589021361;
              } else {
                result[0] += -0.0456575615298775;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                  result[0] += 0.009497883371902986;
                } else {
                  result[0] += 0.02397217641761584;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                  result[0] += 0.0140775945968284;
                } else {
                  result[0] += -0.022956805092882226;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.08672815223341518;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.038405062840459823;
          } else {
            result[0] += -0.03815972199116671;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
        result[0] += 0.11935405197185099;
      } else {
        result[0] += -0.0350275589125087;
      }
    } else {
      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.551017761230469638) ) ) {
        result[0] += -0.00011151180628429768;
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.465643882751465732) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.93885374069213956) ) ) {
                    result[0] += -0.22381388948127334;
                  } else {
                    result[0] += 0.009187105023131647;
                  }
                } else {
                  result[0] += -0.03502475083069856;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
                    result[0] += 0.053757215571297624;
                  } else {
                    result[0] += 0.025339361076157668;
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.03805849860252161;
                    } else {
                      result[0] += 0.012196997127538677;
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.024262553394073294;
                    } else {
                      result[0] += -0.0266443657924722;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.016002847609421597;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.535966873168947089) ) ) {
                  result[0] += -0.013123551676791113;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += -0.0383628501039691;
                  } else {
                    result[0] += -0.10095703319418121;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.914472818374634233) ) ) {
              result[0] += 0.013748762221835588;
            } else {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.014968275001723319;
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.012967946383568255;
                    } else {
                      result[0] += -0.004248271344949355;
                    }
                  }
                } else {
                  result[0] += -0.02935874609286846;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.623839378356934482) ) ) {
                  result[0] += 0.002824388367049667;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.040072342493866354;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.0075254764678944555;
                    } else {
                      result[0] += 0.040115396524744606;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.016506705416855224;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.07761392658445265;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.596743106842042792) ) ) {
                  result[0] += 0.07048768047854409;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                    result[0] += -0.055255527590913046;
                  } else {
                    result[0] += 0.04084369710711352;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
              result[0] += -0.03675985348323601;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.03605495693398158;
              } else {
                result[0] += -0.0026901718907949056;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0007138258920390293;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.00015975343006683834;
          } else {
            result[0] += -0.042276016776209874;
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.094205617904663974) ) ) {
          result[0] += 0.0017385159816016823;
        } else {
          result[0] += -0.016416309500576792;
        }
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.036256466551281716;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.02772457534317346;
              } else {
                result[0] += -0.007071728911283243;
              }
            }
          } else {
            result[0] += -0.050164989854230346;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0056494166354495515;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)16.36023521423340199) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                result[0] += 0.0004855379719658914;
              } else {
                result[0] += -0.01083871878112997;
              }
            } else {
              result[0] += 0.043286483216019384;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.004713000083837222;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.62517356872558771) ) ) {
                result[0] += -0.06560160113381;
              } else {
                result[0] += 0.006088248557146115;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += -0.01315586115033576;
            } else {
              result[0] += 0.012522602911941183;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.1371946815370903;
              } else {
                result[0] += -0.028374735878035653;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.013727045628132746;
                } else {
                  result[0] += 0.09808779617535268;
                }
              } else {
                result[0] += -0.03284897644743109;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.010066671490414528;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.84935188293457209) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += -0.008535042136429055;
                    } else {
                      result[0] += 0.038873346023530236;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.090673208236695224) ) ) {
                      result[0] += -0.06546550464327942;
                    } else {
                      result[0] += 0.11736920897512584;
                    }
                  }
                } else {
                  result[0] += 0.07161263014636564;
                }
              } else {
                result[0] += 0.08659786908734;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.63218307495117365) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.008992828827279175;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01558259728802256;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.017018325516469923;
                  } else {
                    result[0] += -0.05604557112429803;
                  }
                } else {
                  result[0] += 0.014228874602160886;
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.010105123071336378;
              } else {
                result[0] += -0.009844089516958688;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
              result[0] += -0.010754036051373617;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.05216938478031746;
              } else {
                result[0] += -0.009699034226858926;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
          result[0] += 0.0005243735969284915;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.010248730371385198;
            } else {
              result[0] += -0.052884804096944876;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.00967284619343301;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                      result[0] += -0.019703363773117193;
                    } else {
                      result[0] += 0.03097684558277297;
                    }
                  }
                } else {
                  result[0] += -0.042914941907839614;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.02297417622408852;
                  } else {
                    result[0] += -0.031119933536698282;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.09922236332808552;
                  } else {
                    result[0] += 0.004580348770948516;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0025895718754610822;
                  } else {
                    result[0] += 0.12147119534240501;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.00968663195767122;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.04943123889955272;
                    } else {
                      result[0] += -0.002717055963806073;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.0047573094992897965;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                    result[0] += 0.024325260011888103;
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.049685386897606244;
                    } else {
                      result[0] += 0.16262448719769076;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.0046023867580883724;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
          result[0] += -0.021117313239018234;
        } else {
          result[0] += 0.005291257420318173;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.278613805770874912) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.008746354935800628;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.041387319564820224) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.247576236724854404) ) ) {
                    result[0] += 0.01604072779710791;
                  } else {
                    result[0] += -0.05966474214510569;
                  }
                } else {
                  result[0] += -0.010530973140283213;
                }
              } else {
                result[0] += -0.0027708765957026957;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                result[0] += -0.011571370326892273;
              } else {
                result[0] += 0.012288699134618197;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.300811052322388583) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.400584220886231357) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.01217010182567487;
                } else {
                  result[0] += -0.01207933738613138;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.008010823883334219;
                } else {
                  result[0] += -0.04207058999020597;
                }
              }
            } else {
              result[0] += -0.06730982467198687;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.03899060164914997;
            } else {
              result[0] += 0.017570301545462037;
            }
          }
        }
      } else {
        result[0] += 0.005093469140961938;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += -0.00010019234426301385;
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
            result[0] += 0.024195743628946904;
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.056571516579770886;
            } else {
              result[0] += -0.012665770938101638;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.007683503749715371;
              } else {
                result[0] += -0.010618982947657897;
              }
            } else {
              result[0] += 0.005569434878425156;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81821727752685725) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.013552315805532678;
                      } else {
                        result[0] += 0.06362462321522783;
                      }
                    } else {
                      result[0] += 0.08068236223700491;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80468511581421076) ) ) {
                      result[0] += -0.027949719139063273;
                    } else {
                      result[0] += 0.09513367442236415;
                    }
                  }
                } else {
                  result[0] += -0.03852608597473165;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.011104322634004182;
                  } else {
                    result[0] += -0.08388741884096526;
                  }
                } else {
                  result[0] += 0.03324054603375715;
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.05912168048848378;
                } else {
                  result[0] += 0.02074810330695606;
                }
              } else {
                result[0] += 0.01250112739781474;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.000472882520963902;
      } else {
        result[0] += -0.019510239728453758;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.02932894166227505;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.09980443403800948;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.08789571316438959;
                  } else {
                    result[0] += -0.05663113907450602;
                  }
                } else {
                  result[0] += 0.018390530301154544;
                }
              }
            }
          } else {
            result[0] += -0.010932522374938657;
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.06767001670520842;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.0019031678849315962;
              } else {
                result[0] += -0.06289316089046669;
              }
            }
          } else {
            result[0] += 0.006373615014441777;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.665476083755494052) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += 0.016400683271057145;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.010846069173588845;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.05805890439873508;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                      result[0] += -0.010964430474791065;
                    } else {
                      result[0] += -0.03653010786682049;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.022312074940605102;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.015086622403369294;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.03874743435734981;
                        } else {
                          result[0] += 0.03209298665494622;
                        }
                      } else {
                        result[0] += 0.0005007055197472857;
                      }
                    }
                  }
                } else {
                  result[0] += -0.030387547652908534;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += 0.0046896728807927805;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.002928319460198697;
              } else {
                result[0] += 0.12830506641039105;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.801954269409180576) ) ) {
            result[0] += 0.0058958012882682626;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.056690262673066816;
            } else {
              result[0] += 0.08042515259633058;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.278613805770874912) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.0005707111384775779;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
              result[0] += -0.005508816967871622;
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.011804957267802493;
              } else {
                result[0] += -0.05940252130926363;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0007814966978313339;
          } else {
            result[0] += 0.012975097347060795;
          }
        }
      } else {
        result[0] += -0.0312788224811776;
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.0011143757536847783;
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.013987676141697485;
            } else {
              result[0] += -0.04679034543451149;
            }
          } else {
            result[0] += 0.005356180302507823;
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.0025052541863245727;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.02571413203206306;
                  } else {
                    result[0] += -0.0032839017384327903;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
                  result[0] += 0.001786561117324087;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.01169351363670206;
                  } else {
                    result[0] += 0.018935783996350514;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  result[0] += 0.007338515378713721;
                } else {
                  result[0] += -0.020944723208601804;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.41290044784546076) ) ) {
                  result[0] += -0.03656320313791315;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                    result[0] += 0.09457477242369981;
                  } else {
                    result[0] += 0.005843966537582908;
                  }
                }
              }
            }
          } else {
            result[0] += 0.03133534637649239;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.9054608345031756) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0067372655390678104;
                } else {
                  result[0] += -0.030472826387558057;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += -0.01616410010379146;
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.007536020786176921;
                  } else {
                    result[0] += 0.08408898082279895;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                result[0] += -0.03375435297405602;
              } else {
                result[0] += 0.01675551103554464;
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18134641647339045) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.584795951843263495) ) ) {
                          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                              result[0] += 0.007283527112744655;
                            } else {
                              result[0] += -0.023945792688731326;
                            }
                          } else {
                            result[0] += -0.023656530189032167;
                          }
                        } else {
                          result[0] += 0.0669380518155762;
                        }
                      } else {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.016656720593381982;
                        } else {
                          result[0] += -0.05187701197199732;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.06394612227409023;
                      } else {
                        result[0] += 0.02258118166794186;
                      }
                    }
                  } else {
                    result[0] += 0.012698476182536328;
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += 0.011330614485887413;
                    } else {
                      result[0] += -0.022847038111768907;
                    }
                  } else {
                    result[0] += 0.02064588695110066;
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.04054766689732573;
                } else {
                  result[0] += -0.034320863143499275;
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.025192260742188388) ) ) {
                result[0] += -0.033505216257590426;
              } else {
                result[0] += 0.07057179002674764;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.14095449447632014) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
        result[0] += 0.005675143309736657;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
          result[0] += 0.00040033376956461364;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.003066431506691426;
              } else {
                result[0] += -0.01983591741009251;
              }
            } else {
              result[0] += -0.04324714765156695;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += -0.0663783193512796;
              } else {
                result[0] += -0.008314358818306685;
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.018888983524442876;
                      } else {
                        result[0] += -0.018731370788473815;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.023281807342287283;
                      } else {
                        result[0] += -0.0366505158012125;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
                      result[0] += 0.06832896141524351;
                    } else {
                      result[0] += 0.002307978934917744;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.057039357559632645;
                  } else {
                    result[0] += 0.011803847444672291;
                  }
                }
              } else {
                result[0] += -0.03546310537494907;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.010891553216582953;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
        result[0] += -0.0021630166091560995;
      } else {
        result[0] += -0.02787779482333181;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.0012385975733981095;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                result[0] += -0.05985698296458476;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                  result[0] += -0.005370620470000333;
                } else {
                  result[0] += -0.03138107544462015;
                }
              }
            }
          } else {
            result[0] += 0.008471471802420359;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.66775608062744318) ) ) {
                result[0] += -0.0030914384295159386;
              } else {
                result[0] += -0.041362331343172395;
              }
            } else {
              result[0] += -0.05081001609960767;
            }
          } else {
            result[0] += 0.012395066669334103;
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09398412704467951) ) ) {
              result[0] += -0.0006829389390968378;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.011348550890037515;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                  result[0] += -0.012276142973368575;
                } else {
                  result[0] += -0.07688611692743957;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.003294346219406922;
            } else {
              result[0] += 0.008123082502590556;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
            result[0] += 0.003484790077246608;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.004811683234969194;
            } else {
              result[0] += 0.021287885723253852;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
            result[0] += 0.02885932148205675;
          } else {
            result[0] += -0.01817309943301895;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.024238969725506404;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                  result[0] += -0.03672177605194193;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.026905620107674662;
                  } else {
                    result[0] += -0.016802823170775836;
                  }
                }
              } else {
                result[0] += 0.04903578366834027;
              }
            }
          } else {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.017248405425796094;
            } else {
              result[0] += -0.03903902818731428;
            }
          }
        }
      } else {
        result[0] += -0.06614492188460071;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
            result[0] += 0.011647933165813723;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.01679545485938722;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.13223537965820414;
              } else {
                result[0] += -0.011803927415989618;
              }
            }
          }
        } else {
          result[0] += -0.00018823152051694494;
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += 0.00048556019283129055;
              } else {
                result[0] += -0.02926255447282743;
              }
            } else {
              result[0] += 0.08221215173051105;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.07775956546991053;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.008231088123703788;
                  } else {
                    result[0] += 0.07876139720602292;
                  }
                }
              } else {
                result[0] += -0.05324388927961552;
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.01513406311548814;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.04302263914682438;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                      result[0] += -0.0619504127016047;
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05176330002786746;
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.02812860443850282;
                        } else {
                          result[0] += -0.03867453800928875;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.015315739751051423;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.465643882751465732) ) ) {
            result[0] += -0.01262566211735283;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.020181634416142933;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.040797747061351135;
                      } else {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.014301864007764467;
                        } else {
                          result[0] += -0.02392905490694043;
                        }
                      }
                    } else {
                      result[0] += 0.004809241579771512;
                    }
                  } else {
                    result[0] += -0.0486737983609762;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                      result[0] += 0.030181376631337575;
                    } else {
                      result[0] += -0.006171724828229405;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.036145934628088086;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.09270222931872513;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.01846191619934516;
                        } else {
                          result[0] += 0.07459110987434739;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.03863226874640967;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
        result[0] += 0.02353183784728925;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.0032994456482126136;
        } else {
          result[0] += -0.020205710191958923;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.058553933611946156;
            } else {
              result[0] += -0.026965666417994683;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += 0.07521413398179737;
            } else {
              result[0] += 0.010212444974212562;
            }
          }
        } else {
          result[0] += -0.030605485169255355;
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.019993039430297843;
            } else {
              result[0] += -0.0015388220544531822;
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.08833128769944634;
            } else {
              result[0] += 0.013094946376324235;
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
            result[0] += 0.009293674488166245;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0007866791688062503;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.012681718280700988;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += 0.11699538014460338;
                      } else {
                        result[0] += -0.051900497571425945;
                      }
                    }
                  } else {
                    result[0] += 0.03900453672864324;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.0014522818438147337;
                    } else {
                      result[0] += -0.019279261496561548;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0711751576153804;
                    } else {
                      result[0] += -0.008117031031642278;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.40467214584350764) ) ) {
                    result[0] += 0.00330244656638077;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.010249993678761489;
                    } else {
                      result[0] += 0.028639771662561125;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
                  result[0] += 0.003531083839117851;
                } else {
                  result[0] += -0.017645974115528304;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.04741363538248814;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.0005216947535773016;
                    } else {
                      result[0] += 0.1275969295949549;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                    result[0] += 0.0004350697139844754;
                  } else {
                    result[0] += 0.012716147662685924;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.467917680740357333) ) ) {
            result[0] += -0.005825453079577417;
          } else {
            result[0] += 0.03602259861668706;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            result[0] += 0.02545364097702446;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.029456101795104307;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.01782154421703077;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.030829764183873787;
                  } else {
                    result[0] += -0.037302742150848466;
                  }
                } else {
                  result[0] += -0.02130742370037149;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.044934975110656666;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.0009374256954209685;
        } else {
          result[0] += -0.01719005411271719;
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += 0.0001911247459722303;
              } else {
                result[0] += -0.026293168785292823;
              }
            } else {
              result[0] += 0.0759795472471628;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03983088392316983;
              } else {
                result[0] += -0.05610488025709051;
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.015425100801765929;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.03731083782467951;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.07833033635358044;
                        } else {
                          result[0] += -0.03482769985911429;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                          result[0] += -0.04170425442563419;
                        } else {
                          result[0] += 0.0002734236846676064;
                        }
                      }
                    } else {
                      result[0] += -0.0768811815161905;
                    }
                  } else {
                    result[0] += 0.012781719521849678;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0015879053207089296;
              } else {
                result[0] += -0.02184401823854215;
              }
            } else {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.006804077667800664;
                } else {
                  result[0] += 0.07324191762969438;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                  result[0] += 0.003779627749329026;
                } else {
                  result[0] += -0.02449049954024056;
                }
              }
            }
          } else {
            result[0] += -0.02143241198450277;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.0821280343211959;
        } else {
          result[0] += 0.010726214353843501;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
          result[0] += -0.001652679587895636;
        } else {
          result[0] += -0.012055409523331971;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.215607166290284091) ) ) {
                  result[0] += -0.038039724439117226;
                } else {
                  result[0] += -0.1516467059351651;
                }
              } else {
                result[0] += 0.06255028864821782;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.10302565453925416;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.024890070730243;
                  } else {
                    result[0] += -0.012213590635227124;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.198464870452881303) ) ) {
                    result[0] += -0.05025118492948323;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.004770899011384777;
                    } else {
                      result[0] += -0.017502974074230078;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0062309805371220175;
            } else {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.013329524030782958;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.005732218420084003;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.020266164689753837;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.494223117828370029) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                                  result[0] += -0.014216611934506074;
                                } else {
                                  result[0] += -0.15169059453215386;
                                }
                              } else {
                                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                                  result[0] += -0.08026794920957599;
                                } else {
                                  result[0] += 0.07610582712118588;
                                }
                              }
                            } else {
                              result[0] += 0.06946857756072872;
                            }
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                              result[0] += -0.08443287659011701;
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                                result[0] += 0.0991123440778009;
                              } else {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                                  result[0] += -0.09803593182630829;
                                } else {
                                  result[0] += -0.022990035327063434;
                                }
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.397998809814454013) ) ) {
                          result[0] += -0.04218121722914203;
                        } else {
                          result[0] += 0.04790561450545832;
                        }
                      }
                    } else {
                      result[0] += 0.08714627534754482;
                    }
                  } else {
                    result[0] += -0.0023959800931814914;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.0009715625735485585;
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.02604460716247603) ) ) {
                result[0] += -0.049313502412175914;
              } else {
                result[0] += 0.018615656520718473;
              }
            } else {
              result[0] += -0.012303873891711654;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.898905277252199042) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  result[0] += -0.056705607961831994;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.04834099144153794;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                          result[0] += 0.09829263423087126;
                        } else {
                          result[0] += 0.016904469223715118;
                        }
                      } else {
                        result[0] += -0.001115618603960997;
                      }
                    }
                  } else {
                    result[0] += -0.06745589111988369;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                  result[0] += -0.12378069014870541;
                } else {
                  result[0] += -0.05097092942959755;
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                result[0] += 0.02082978713706675;
              } else {
                result[0] += -0.04084527703280598;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                  result[0] += 0.005578337513873661;
                } else {
                  result[0] += -0.020668223856364843;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                  result[0] += 0.0058962421766321875;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.02744227913141671;
                  } else {
                    result[0] += 0.011176038280752245;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.535966873168947089) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.11474648689649487;
                  } else {
                    result[0] += 0.002078064613259529;
                  }
                } else {
                  result[0] += 0.007460301579660643;
                }
              } else {
                result[0] += 0.009501618252402345;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0035713582239550916;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                  result[0] += 0.00445186508334594;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0005793216333344624;
                      } else {
                        result[0] += -0.04107882348369038;
                      }
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += -0.040725699734202384;
                      } else {
                        result[0] += -0.01113843694824533;
                      }
                    }
                  } else {
                    result[0] += 0.0071157334287779905;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  result[0] += 0.0026681980917605173;
                } else {
                  result[0] += -0.039194303083242066;
                }
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.001574832651829419;
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += 0.0032248179374324893;
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                  result[0] += -0.021993682976794493;
                } else {
                  result[0] += 0.003580628178958363;
                }
              } else {
                result[0] += -0.04432437613616931;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0009067438324141839;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                      result[0] += -0.006033703869303697;
                    } else {
                      result[0] += -0.03219344260239917;
                    }
                  } else {
                    result[0] += -0.046930241505595334;
                  }
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.003357499271740152;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2121162414550799) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.08624171583737322;
                        } else {
                          result[0] += -0.0015720398474245029;
                        }
                      } else {
                        result[0] += 0.045445574711753915;
                      }
                    }
                  } else {
                    result[0] += -0.014991597551639747;
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.012904930834804741;
                  } else {
                    result[0] += -0.017766720921443928;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.008010084815869189;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += -0.007791012661057172;
                } else {
                  result[0] += 0.03209286394204713;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.01725792884826749) ) ) {
                  result[0] += -0.04694164970287173;
                } else {
                  result[0] += 0.116490253119729;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.058601807925148;
          } else {
            result[0] += -0.01818612304156609;
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.95797300338745206) ) ) {
            result[0] += 0.0027746186579156178;
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.005882536153402399;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.005320342679432451;
              } else {
                result[0] += 0.031118484623603927;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.011491144849059314;
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += 0.011785664712575526;
          } else {
            result[0] += -0.022379355793630998;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            result[0] += 0.026289275560464787;
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.00856763383536727;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.09203437492813853;
                      } else {
                        result[0] += 0.09087875853381146;
                      }
                    } else {
                      result[0] += 0.016098527988064305;
                    }
                  } else {
                    result[0] += -0.01032815534905658;
                  }
                }
              } else {
                result[0] += -0.021020978649957972;
              }
            } else {
              result[0] += -0.03756288281645784;
            }
          }
        }
      } else {
        result[0] += -0.061387819319297224;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
        result[0] += 0.0005674026516309946;
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                result[0] += -0.00037025222614481387;
              } else {
                result[0] += -0.024584681052571992;
              }
            } else {
              result[0] += 0.0706448234802549;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.04075514374613601;
              } else {
                result[0] += 0.032490245014552985;
              }
            } else {
              result[0] += -0.02518291062648917;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.705447435379029208) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.007075280351348791;
                } else {
                  result[0] += -0.019642490423334883;
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.009886158439449866;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.0296510384738097;
                    } else {
                      result[0] += -0.06665617104909542;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.025393914364190168;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.009236138223269987;
                    } else {
                      result[0] += -0.06572890185982172;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.012135603901717558;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.012381250076627121;
                } else {
                  result[0] += -0.02373852813296038;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += 0.008040587941613633;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.07854447377602858;
                  } else {
                    result[0] += 0.10730790639496648;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.005080838958172761;
                } else {
                  result[0] += -0.021048375738633798;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.02400266803766478;
                } else {
                  result[0] += 0.0312562499585525;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.00024688759385451336;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += -0.007010050549580715;
          } else {
            result[0] += -0.026914620808342907;
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.24121904373169123) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.005481757681405028;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += -0.11746445964852359;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += 0.037574571005341076;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.017797946929933417) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.006775749972249829;
                      } else {
                        result[0] += -0.037914694150836;
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.012837220845722459;
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                                result[0] += -0.020351637715655863;
                              } else {
                                result[0] += -0.06236860358924983;
                              }
                            } else {
                              result[0] += 0.006749562954720891;
                            }
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                              result[0] += 0.00486728784227548;
                            } else {
                              result[0] += -0.048480763662188356;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.004443529664254638;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.007273803225519006;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.05268940110714965;
              } else {
                result[0] += -0.016529674382709792;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.947025299072267401) ) ) {
                result[0] += -0.06925846132157797;
              } else {
                result[0] += 0.0675363103211338;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += -0.008846569816886075;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.015635154823923653;
              } else {
                result[0] += -9.750734890830419e-05;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
                result[0] += 0.0037204473120977;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += -0.01842810783157099;
                    } else {
                      result[0] += -0.07235591674207255;
                    }
                  } else {
                    result[0] += -0.002943483522665014;
                  }
                } else {
                  result[0] += -0.05025016130238075;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.289595603942871982) ) ) {
                    result[0] += -0.005298145518065942;
                  } else {
                    result[0] += 0.01046672972159489;
                  }
                } else {
                  result[0] += -0.03408197773758091;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                    result[0] += 0.015934614036148938;
                  } else {
                    result[0] += -0.05176174682661271;
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.026406103683717527;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.012926784537276186;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            result[0] += 0.0666430152348386;
                          } else {
                            result[0] += 0.014909117267021273;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.02108549792680768;
                      } else {
                        result[0] += 0.09583857390241429;
                      }
                    }
                  } else {
                    result[0] += 0.006390541056765013;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.012271155996488167;
        } else {
          result[0] += -0.010203756394395469;
        }
      } else {
        result[0] += -0.06481180218405053;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
          result[0] += -0.004908429328690971;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.006832038271872672;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.018707278069166444;
            } else {
              result[0] += 0.001449051481122868;
            }
          }
        }
      } else {
        result[0] += -0.009753768568826308;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
          result[0] += 0.022881777329065357;
        } else {
          result[0] += -0.01128537285208648;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
          result[0] += -0.022586870747535143;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.01293420791626154) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
                result[0] += 0.016687579740945733;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                  result[0] += -0.003330319913869357;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.025985366462469274;
                  } else {
                    result[0] += 0.03925602370790074;
                  }
                }
              }
            } else {
              result[0] += -0.028284318154566657;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.008167419650008929;
                } else {
                  result[0] += 0.08875125580186796;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.0181190786928865;
                } else {
                  result[0] += -0.04341212725569081;
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.004506209036982565;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0066064751735428265;
                } else {
                  result[0] += 0.13888987619294854;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
            result[0] += 0.0023069067580240646;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
              result[0] += -0.02679452454501334;
            } else {
              result[0] += -0.0023232688125471314;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.0034558729952058834;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.02278078715640869;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.05536564204403343;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.025423079191724214;
                      } else {
                        result[0] += -0.05588496263657553;
                      }
                    } else {
                      result[0] += 0.0007113161727093717;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.03287457031241758;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.55604696273803889) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += 0.046909084655561525;
                      } else {
                        result[0] += -0.06724014669160987;
                      }
                    } else {
                      result[0] += -0.061951285359657896;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += 0.041918441393515374;
                      } else {
                        result[0] += -0.07311162949885278;
                      }
                    } else {
                      result[0] += -0.04729390496291048;
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.017440372011627608;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.2807660102844256) ) ) {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.10092760999767857;
                        } else {
                          result[0] += -0.03207987341349277;
                        }
                      } else {
                        result[0] += 0.04397452141784181;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.015273494864647998;
                      } else {
                        result[0] += 0.015246499764381189;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.04945590198548579;
                      } else {
                        result[0] += -0.02382763206937175;
                      }
                    }
                  } else {
                    result[0] += 0.026815376179058655;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.59565925598144709) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.042536703613603594;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
                        result[0] += 0.19377977870189467;
                      } else {
                        result[0] += 0.028053675519611694;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                      result[0] += -0.024096203681306362;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += 0.025833267086380854;
                          } else {
                            result[0] += -0.03252120658982639;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += -0.03149976094300225;
                          } else {
                            result[0] += 0.08209285782914384;
                          }
                        }
                      } else {
                        result[0] += 0.06567937692197201;
                      }
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.048191320851595165;
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.014199864574377245;
        } else {
          result[0] += -0.020138133394994415;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
        result[0] += 0.0003893358711735302;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += -0.01620757329214798;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0011389448370337618;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.008410329939356942;
                  } else {
                    result[0] += -0.036473341911080155;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.0335511659422559;
                    } else {
                      result[0] += 0.03169617432684934;
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.01861148391079055;
                    } else {
                      result[0] += 0.013948781052416453;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.010885184318613456;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += 0.01971303645898228;
                    } else {
                      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.061106809498681094;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.07819769397565596;
                        } else {
                          result[0] += 0.018585153229375796;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.016069748376572195;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.00236139251999485;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += -0.033288384001872555;
                    } else {
                      result[0] += 0.00539789720172448;
                    }
                  } else {
                    result[0] += 0.02613842490409514;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
            result[0] += -0.02954994943830294;
          } else {
            result[0] += 0.016359484100605098;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
        result[0] += 0.027766011241893762;
      } else {
        result[0] += -0.03502708295615189;
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.06542849495161875;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          result[0] += 0.035951223838805606;
        } else {
          result[0] += -0.030586485105885075;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.0005923966099612729;
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
        result[0] += -0.02070248388805139;
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += -0.002064731558545052;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.005893676639647757;
                      } else {
                        result[0] += 0.02281427094780051;
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.01752880201508278;
                      } else {
                        result[0] += -0.02272807783580401;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += -0.01584544397109069;
                      } else {
                        result[0] += 0.04189396132453635;
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                          result[0] += -0.01503793866628407;
                        } else {
                          result[0] += -0.062366686538026773;
                        }
                      } else {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.01872111899450954;
                        } else {
                          result[0] += -0.0362205469176221;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.025285808430834462;
                }
              }
            } else {
              result[0] += -0.02300337752359792;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.028737822556656214;
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.04799740772857623;
                      } else {
                        result[0] += 0.010239273518621915;
                      }
                    }
                  } else {
                    result[0] += -0.08516464093307176;
                  }
                } else {
                  result[0] += 0.005925028140568342;
                }
              } else {
                result[0] += -0.10492651965072955;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.017887801687289717;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                    result[0] += 0.06322290617344105;
                  } else {
                    result[0] += 0.004286484715886006;
                  }
                }
              } else {
                result[0] += -0.005507464167161144;
              }
            }
          }
        } else {
          result[0] += -0.053093857016879936;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += 0.01563731732931948;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.029471676346911353;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += 0.10402594732350481;
                    } else {
                      result[0] += -0.017282962847017153;
                    }
                  } else {
                    result[0] += -0.022347313387461064;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.020098536249446942;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += 0.028693747591978333;
                    } else {
                      result[0] += -0.005420485450099984;
                    }
                  } else {
                    result[0] += -0.019810642269654847;
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.0031112303202006455;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.384830474853516513) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.021427265856285543;
                          } else {
                            result[0] += -0.042044548116313335;
                          }
                        } else {
                          result[0] += 0.02535663767287576;
                        }
                      } else {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += 0.005371710577162589;
                            } else {
                              result[0] += 0.07467428579339007;
                            }
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                              result[0] += 0.028426231357025752;
                            } else {
                              result[0] += -0.07103341319516591;
                            }
                          }
                        } else {
                          result[0] += 0.05838353022947621;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.028868357711316503;
                    } else {
                      result[0] += 0.06772268318543759;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.044303301734059226;
            } else {
              result[0] += 0.03067814352947755;
            }
          }
        } else {
          result[0] += -0.023158014278563732;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.04377134337788758;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.019150283440400404;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.05226964530095438;
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.045566769806362574;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += -0.026895844702287747;
                      } else {
                        result[0] += 0.041690430085446754;
                      }
                    } else {
                      result[0] += -0.037472666982513024;
                    }
                  }
                } else {
                  result[0] += -0.05616840183771176;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.030364224765268427;
                } else {
                  result[0] += 0.006760889604830549;
                }
              } else {
                result[0] += -0.051574303782184694;
              }
            } else {
              result[0] += 0.012676204706961347;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.06407229926760097;
            } else {
              result[0] += 0.08234413169927617;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
        result[0] += -0.0022404279851296867;
      } else {
        result[0] += 0.032617811027756675;
      }
    } else {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              result[0] += 0.0060274294217417795;
            } else {
              result[0] += -0.053947847842150634;
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.026579158519959453;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                  result[0] += -0.043682459616800395;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.028102322962325843;
                  } else {
                    result[0] += 0.10261361057033294;
                  }
                }
              }
            } else {
              result[0] += -0.03006631599849942;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.002582878255132235;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.0013022109923448099;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80060577392578303) ) ) {
                    result[0] += 0.021984463348872732;
                  } else {
                    result[0] += -0.04233665857433877;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                    result[0] += 0.003598521532548209;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.016435680846912857;
                    } else {
                      result[0] += -0.0484505937819375;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.417592287063599077) ) ) {
              result[0] += -0.0006417584237432292;
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.02199725820977382;
                    } else {
                      result[0] += 0.001226773200057426;
                    }
                  } else {
                    result[0] += -0.021194434198406043;
                  }
                } else {
                  result[0] += 0.002690040907764273;
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.017212753664268353;
                  } else {
                    result[0] += 0.09078859526941352;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.007851536353653292;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.2807660102844256) ) ) {
                          result[0] += -0.05934036234553394;
                        } else {
                          result[0] += -0.006972099332739308;
                        }
                      } else {
                        result[0] += 0.001449527891119536;
                      }
                    }
                  } else {
                    result[0] += -0.05704841376158844;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.542080402374269354) ) ) {
                result[0] += 0.030301622684647894;
              } else {
                result[0] += -0.08797740843039709;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.023644614939313937;
                } else {
                  result[0] += -0.05811685813440912;
                }
              } else {
                result[0] += -0.07941423212754752;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.25862455368042081) ) ) {
                result[0] += 0.060359950582346146;
              } else {
                result[0] += -0.03362933547996309;
              }
            } else {
              result[0] += 0.001387728447292685;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.011906735206994454;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.009341904341137816;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.825422286987305576) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
                        result[0] += 0.07533544572916841;
                      } else {
                        result[0] += 0.00507811707852367;
                      }
                    } else {
                      result[0] += 0.09401160569304091;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                  result[0] += 0.025295922764533137;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.061095545127280396;
                    } else {
                      result[0] += 0.007851166311686699;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      result[0] += -0.017314532524172116;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                              result[0] += 0.06655580440398866;
                            } else {
                              result[0] += -0.009507498631899402;
                            }
                          } else {
                            result[0] += 0.0647525857662779;
                          }
                        } else {
                          result[0] += 0.08166941901654384;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.041387319564820224) ) ) {
                          result[0] += 0.09958786995593248;
                        } else {
                          result[0] += -0.03809944288394341;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
                    result[0] += 0.0017348954760557527;
                  } else {
                    result[0] += -0.06795958751590635;
                  }
                } else {
                  result[0] += -0.10954808812000419;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
                  result[0] += 0.014828041349751723;
                } else {
                  result[0] += -0.03352303212588214;
                }
              }
            }
          } else {
            result[0] += -0.02820694191739935;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.655405282974244052) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.124530076980591708) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
            result[0] += -0.05172739686268157;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += 0.07204087731186744;
            } else {
              result[0] += 0.00020165127319645977;
            }
          }
        } else {
          result[0] += -0.04398433563019827;
        }
      } else {
        result[0] += 0.1529498953395936;
      }
    } else {
      result[0] += -0.0745148404527025;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0019756539340958575;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.03271809653494576;
              } else {
                result[0] += -0.006434146257187873;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0004175157822617618;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.003515763485310826;
                  } else {
                    result[0] += -0.04881918762255443;
                  }
                } else {
                  result[0] += 0.006790613458369805;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += -0.023483636736389722;
                    } else {
                      result[0] += -0.11333420569559019;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.0012369138746770002;
                    } else {
                      result[0] += 0.18558429758842238;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.01788058253971289;
                    } else {
                      result[0] += 0.007054595171041874;
                    }
                  } else {
                    result[0] += 0.033036434626031064;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0034535915741490085;
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.02523688904923571;
                } else {
                  result[0] += -0.0626571984092849;
                }
              } else {
                result[0] += 0.0016222292190392713;
              }
            } else {
              result[0] += 0.03691023606800398;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.005352873122833925;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.0019733765360643025;
              } else {
                result[0] += -0.024683660786443986;
              }
            } else {
              result[0] += 0.009312859543419387;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
            result[0] += -0.012557665282456982;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.06568754262548361;
            } else {
              result[0] += -0.004990897344373927;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.02515548497788829;
          } else {
            result[0] += 0.10573487201107792;
          }
        } else {
          result[0] += 0.01150278233570226;
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.00897623444856054;
        } else {
          result[0] += -0.05564540289753586;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.009234109141611492;
        } else {
          result[0] += 0.013169844125176668;
        }
      } else {
        result[0] += -7.731956847335209e-05;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.25333833694458185) ) ) {
              result[0] += 0.012738165751468512;
            } else {
              result[0] += -0.011213647027466994;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.03880758440081519;
                } else {
                  result[0] += -0.00445944769701895;
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.959391355514527255) ) ) {
                      result[0] += 0.006188267372939084;
                    } else {
                      result[0] += -0.06569476359842426;
                    }
                  } else {
                    result[0] += -0.04032988911542934;
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.01393017208379297;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
                      result[0] += 0.022688167756709156;
                    } else {
                      result[0] += -0.04350208135216965;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.044687379075865216;
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.009077439883892207;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0576403186063191;
                    } else {
                      result[0] += -0.028896579696296977;
                    }
                  } else {
                    result[0] += 0.007551948930084176;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.01522833701167997;
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.03606373154729646;
                    } else {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.06117560884679546;
                      } else {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.012485422238228896;
                        } else {
                          result[0] += -0.06620006624143208;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += -0.00010232004756814927;
                  } else {
                    result[0] += -0.03251670924561585;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.004528289028031088;
              } else {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.026829896155182605;
                } else {
                  result[0] += 0.09944743711390748;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.28299736976623624) ) ) {
              result[0] += 0.002499547327607963;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.007616962065619532;
              } else {
                result[0] += 0.07988590958018577;
              }
            }
          }
        }
      } else {
        result[0] += -0.05991738230290783;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
        result[0] += 0.013111365529376305;
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.007033705935718407;
        } else {
          result[0] += -0.00655359129644587;
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.018862705722538193;
              } else {
                result[0] += 0.00584128830426199;
              }
            } else {
              result[0] += 0.00690737875493425;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
                  result[0] += 0.10830511586402708;
                } else {
                  result[0] += -0.02831908617121126;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.105651378631592685) ) ) {
                  result[0] += 0.014280631901670235;
                } else {
                  result[0] += -0.011978443060002505;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.02050998718544103;
              } else {
                result[0] += -0.06643835111260833;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.005344058248996558;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                result[0] += 0.066868615678442;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.07151103856941693;
                } else {
                  result[0] += 0.011775154566191683;
                }
              }
            }
          } else {
            result[0] += 0.03147520691891907;
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.623839378356934482) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.008305375952328385;
            } else {
              result[0] += 0.011986888984690145;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.06744219625984944;
              } else {
                result[0] += -0.018774640827691095;
              }
            } else {
              result[0] += -0.07042964863926898;
            }
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.230628252029419833) ) ) {
                result[0] += -0.003623277159418025;
              } else {
                result[0] += -0.04915134728201458;
              }
            } else {
              result[0] += -0.03083327320208311;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += 0.0059266047940055725;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.029291003705836288;
                      } else {
                        result[0] += 0.011665547687546523;
                      }
                    } else {
                      result[0] += 0.02421790326098574;
                    }
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.004814036846368489;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.06885140629942754;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                            result[0] += -0.03073265660091877;
                          } else {
                            result[0] += 0.017519418202001865;
                          }
                        } else {
                          result[0] += 0.1682909905140026;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.03200228722435424;
                    } else {
                      result[0] += 0.1278964612870879;
                    }
                  } else {
                    result[0] += -0.09158925956731209;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += 0.0002856379280942128;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.007473032778887243;
                  } else {
                    result[0] += 0.008762420661371987;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
          result[0] += 0.002807111376969889;
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.992712974548340732) ) ) {
            result[0] += -0.0024673334897498634;
          } else {
            result[0] += -0.025353975029979194;
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.001327277452308006;
        } else {
          result[0] += -0.03974712799270552;
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.49241352081299006) ) ) {
            result[0] += 0.019831449831919055;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.016273105735230834;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.025576338999778736;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.007057256551493703;
                  } else {
                    result[0] += 0.11192034027982428;
                  }
                }
              } else {
                result[0] += -0.13830992794934036;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += -0.020893824234812862;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
                result[0] += 0.0019466320344725598;
              } else {
                result[0] += -0.01932020612747813;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.012722635846075567;
                    } else {
                      result[0] += -0.057339343660990985;
                    }
                  } else {
                    result[0] += -0.04761978426906107;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.0138128467337744;
                  } else {
                    result[0] += 0.01721981212736606;
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.003476855514341469;
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.016910407857914016;
                  } else {
                    result[0] += 0.12202058404591376;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.03548115107008117;
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += 0.01323785718027633;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09398412704467951) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
            result[0] += 0.001600088225457834;
          } else {
            result[0] += -0.009795127057757272;
          }
        } else {
          result[0] += -0.01247495535306606;
        }
      }
    } else {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
              result[0] += 0.026902346970597416;
            } else {
              result[0] += 0.0012814907257248517;
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.01940667276649147;
                  } else {
                    result[0] += -0.0007728643062551646;
                  }
                } else {
                  result[0] += -0.017270672996470112;
                }
              } else {
                result[0] += 0.007349990891390519;
              }
            } else {
              result[0] += -0.03787747200981893;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.12154971792407848;
            } else {
              result[0] += -0.0020175267952028757;
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.09909164461492459;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.01889182195042077;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
                    result[0] += -0.0434597686793774;
                  } else {
                    result[0] += -0.0018546050667835304;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.008718944696927812;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += 0.013086552463481253;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.009504320588842805;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.582417964935304511) ) ) {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.947818994522095615) ) ) {
                                  result[0] += -0.12378216355231786;
                                } else {
                                  result[0] += -0.05115030070637866;
                                }
                              } else {
                                result[0] += 0.09476278862857757;
                              }
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                                result[0] += -0.0765212063123718;
                              } else {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                                  result[0] += 0.09298697385005134;
                                } else {
                                  result[0] += -0.045701866766715046;
                                }
                              }
                            }
                          } else {
                            result[0] += 0.036217157579896485;
                          }
                        }
                      } else {
                        result[0] += 0.00026197138326815566;
                      }
                    } else {
                      result[0] += 0.0027828707128324153;
                    }
                  } else {
                    result[0] += -0.001026878308814535;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.012572725632873958;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                    result[0] += -0.1865923129319782;
                  } else {
                    result[0] += -0.027670675274499998;
                  }
                } else {
                  result[0] += 0.019182815808206428;
                }
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.010202083667851696;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.1315285989914526;
                  } else {
                    result[0] += -0.03434983261364692;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
                        result[0] += -0.012033330762043987;
                      } else {
                        result[0] += -0.07844172360407634;
                      }
                    } else {
                      result[0] += 0.02304696425114923;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.55753517150879084) ) ) {
                      result[0] += 0.010055273990662026;
                    } else {
                      result[0] += 0.0395008397534619;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.013534884805694553;
                    } else {
                      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.08701832354827757;
                      } else {
                        result[0] += -0.06750032263963639;
                      }
                    }
                  } else {
                    result[0] += -0.0004363093010613794;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                result[0] += -0.07556935785759657;
              } else {
                result[0] += 0.1907631979116649;
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.006429254307254818;
              } else {
                result[0] += -0.09007144183857367;
              }
            }
          }
        } else {
          result[0] += 0.0032563467946105486;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        result[0] += 0.06366376285814816;
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.003919793365060871;
        } else {
          result[0] += -0.03178318620441008;
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.441542863845826083) ) ) {
          result[0] += 0.021807657793256954;
        } else {
          result[0] += 0.0040903313667973095;
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.04946259560845524;
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
            result[0] += -0.0018208915773554896;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += 0.0018195009311094688;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03728845011433221;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                    result[0] += -0.055112408074727795;
                  } else {
                    result[0] += 0.025079835492921226;
                  }
                }
              }
            } else {
              result[0] += -0.0279718729327611;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[40].missing != -1) && (data[40].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.01248077873439896;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.44831323623657404) ) ) {
            result[0] += -0.012867489151607549;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.06403401777984812;
              } else {
                result[0] += -0.0837143231410391;
              }
            } else {
              result[0] += -0.1896351431543936;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.95386886596679865) ) ) {
          result[0] += 0.006219012926015517;
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.02014812246603395;
          } else {
            result[0] += 0.06282749215007556;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.353313446044923651) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                  result[0] += 0.025862644788412183;
                } else {
                  result[0] += 0.12442287112127888;
                }
              } else {
                result[0] += 0.11898942325920901;
              }
            } else {
              result[0] += -0.009688690544708084;
            }
          } else {
            result[0] += -0.005865440327349264;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                result[0] += 0.009326861653877751;
              } else {
                result[0] += -0.09455938598457123;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                result[0] += -0.027098309431590657;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.09013388270427863;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.12000065582380862;
                    } else {
                      result[0] += -0.016045110713496756;
                    }
                  } else {
                    result[0] += -0.06085605894566623;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.960767745971680132) ) ) {
              result[0] += 0.0037879276505336217;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.620046615600586826) ) ) {
                    result[0] += -0.23872273820831258;
                  } else {
                    result[0] += 0.020768128544819603;
                  }
                } else {
                  result[0] += -0.00905317817820034;
                }
              } else {
                result[0] += -0.1804324210296443;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          result[0] += 0.011086910023581097;
        } else {
          result[0] += -0.06251616469945463;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.007397847784563792;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.032860266443809724;
                } else {
                  result[0] += -0.03070082107364105;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                  result[0] += -0.028911458302292883;
                } else {
                  result[0] += -0.10445067827108555;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
              result[0] += 0.022887558140743818;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += 0.040943860394053615;
                } else {
                  result[0] += -0.014392277323427238;
                }
              } else {
                result[0] += -0.05702346012363271;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.737603187561036044) ) ) {
                  result[0] += 0.029204533099236048;
                } else {
                  result[0] += 0.19670866768580264;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.993164777755738193) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.03320408158690875;
                  } else {
                    result[0] += 0.012142699791214167;
                  }
                } else {
                  result[0] += -0.025294387528061693;
                }
              }
            } else {
              result[0] += -0.020366744360972775;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += -0.016031085918547124;
              } else {
                result[0] += 0.023740764812950826;
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.07005769831007355;
                } else {
                  result[0] += -0.01010525486581655;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                    result[0] += -0.046367335159119304;
                  } else {
                    result[0] += -0.09887137740420136;
                  }
                } else {
                  result[0] += 0.15474343968948182;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.06968835301353289;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.733271598815919745) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.680161952972413886) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.00492320184016232;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.025012658120272976;
                    } else {
                      result[0] += 0.007146552996702627;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.15766876480132386;
                  } else {
                    result[0] += -0.0010279185655085786;
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.182065486907959873) ) ) {
                    result[0] += -0.028892717324860713;
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.016468483165082037;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                        result[0] += -0.03552009581763244;
                      } else {
                        result[0] += 0.11606280930952453;
                      }
                    }
                  }
                } else {
                  result[0] += -0.004022346279506243;
                }
              }
            } else {
              result[0] += -0.09535013722029603;
            }
          } else {
            result[0] += -0.03279793259887855;
          }
        }
      }
    } else {
      result[0] += 0.00019209488219563438;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.019866525261247176;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
              result[0] += 0.054702496701891826;
            } else {
              result[0] += -0.005864728417955005;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
            result[0] += -0.009763961219926405;
          } else {
            result[0] += 0.001755713892947436;
          }
        }
      } else {
        result[0] += -0.013488490889028051;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          result[0] += 0.015636740067015507;
        } else {
          result[0] += -0.005220918683243487;
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.008672069245530099;
          } else {
            result[0] += 0.02367854653304274;
          }
        } else {
          result[0] += -0.007833154599413267;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.0008126419275655989;
      } else {
        result[0] += -0.01876861145315231;
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += -0.047692745720989776;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.06111540486725358;
                } else {
                  result[0] += 0.02024290799852791;
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                result[0] += -0.006977545722910096;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.004895306102284579;
                } else {
                  result[0] += -0.05413480142791907;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.1383510991379029;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.05770030526221348;
                } else {
                  result[0] += -0.03801670484904909;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.602003335952759233) ) ) {
                    result[0] += -0.022186961306877966;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.433569431304932529) ) ) {
                      result[0] += 0.03021450075222127;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.005671011791981727;
                      } else {
                        result[0] += 0.02516630677144991;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.801954269409180576) ) ) {
                      result[0] += -0.06437716601962996;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.017119676957820497;
                      } else {
                        result[0] += -0.06378542951233677;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.0419893163667756;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.030871413717439344;
                        } else {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.006061960654244202;
                          } else {
                            result[0] += -0.07242655133361253;
                          }
                        }
                      } else {
                        result[0] += 0.12733841210787486;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                  result[0] += 0.013108711940901328;
                } else {
                  result[0] += -0.03454814424461137;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                  result[0] += 0.09158517269605954;
                } else {
                  result[0] += -0.027702634357721273;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.0007278386213812537;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.019555391791317406;
                  } else {
                    result[0] += 0.07234789412563361;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.03744363610989112;
                    } else {
                      result[0] += 0.09692295655495703;
                    }
                  } else {
                    result[0] += -0.011520045006196196;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.00517870127864792;
                      } else {
                        result[0] += 0.09382333873450535;
                      }
                    } else {
                      result[0] += -0.029307291720497577;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                          result[0] += -0.06965654380737965;
                        } else {
                          result[0] += 0.07554946233465903;
                        }
                      } else {
                        result[0] += 0.10064419500423923;
                      }
                    } else {
                      result[0] += -0.0469503822330766;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.013029821281749449;
                    } else {
                      result[0] += -0.053396051006940386;
                    }
                  } else {
                    result[0] += 0.019275463126953607;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.016588828316847704;
                      } else {
                        result[0] += -0.0203441862869494;
                      }
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.05620587495049239;
                      } else {
                        result[0] += 0.11124136539699243;
                      }
                    }
                  } else {
                    result[0] += -0.016583294477889348;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
              result[0] += 0.0019882758719230883;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0016589346146471792;
              } else {
                result[0] += 0.056421258783119654;
              }
            }
          }
        }
      } else {
        result[0] += -0.051569233921837224;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.002877800851395702;
            } else {
              result[0] += 0.012807024251369923;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.119004011154175693) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.005991955647656299;
              } else {
                result[0] += -0.052497556596088284;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
                result[0] += -0.03830204575597135;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.668133974075318271) ) ) {
                  result[0] += -0.29380300183622554;
                } else {
                  result[0] += 0.007143297124597362;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.634540319442749912) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.011278838455529152;
              } else {
                result[0] += -0.028952788571357736;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.04934036200568317;
              } else {
                result[0] += -0.012835554701285334;
              }
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += -0.017017298140534837;
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.09036369656957782;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += -0.018536993088511415;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.007610046545954098;
                    } else {
                      result[0] += -0.0072467515257737345;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.004417384484683674;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.002603911724319128;
                          } else {
                            result[0] += -0.021138126583924277;
                          }
                        } else {
                          result[0] += -0.041167746952431875;
                        }
                      } else {
                        result[0] += -0.06127025047897209;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.83939445018768355) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.023091812507431007;
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.914472818374634233) ) ) {
                              result[0] += 0.013795328629559409;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += -0.0030654547683871443;
                              } else {
                                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                                  result[0] += -0.010848931202522243;
                                } else {
                                  result[0] += -0.04532143207218115;
                                }
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.009941341404969613;
                            } else {
                              result[0] += 0.00032045780514524086;
                            }
                          } else {
                            result[0] += -0.008136688899549311;
                          }
                        }
                      } else {
                        result[0] += 0.013055300860005887;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.002698674769177403;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                              result[0] += -0.1358530168105718;
                            } else {
                              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += -0.0003814095797782084;
                              } else {
                                result[0] += 0.037311350542945244;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.0574529888899331;
                          } else {
                            result[0] += 0.011665893933297557;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.013699932355875109;
                        } else {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.11123118127159853;
                          } else {
                            result[0] += 0.019643486527897467;
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
        result[0] += -0.012757938373727165;
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.02604460716247603) ) ) {
          result[0] += -0.0020584737900447014;
        } else {
          result[0] += 0.059606526797681816;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.994480729103088823) ) ) {
          result[0] += -0.011659927279514112;
        } else {
          result[0] += 0.012985382519015968;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += 0.012719875870453048;
          } else {
            result[0] += -0.021741541054722906;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += 0.016259143488681346;
            } else {
              result[0] += -0.012746759660739968;
            }
          } else {
            if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.030466947580508937;
            } else {
              result[0] += -0.006369667318644173;
            }
          }
        }
      } else {
        result[0] += -0.05669532185366989;
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.006330886016341158;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
              result[0] += 0.0030852312901655126;
            } else {
              result[0] += -0.0835303193757635;
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.772694945335388628) ) ) {
            result[0] += 0.00022924171423775112;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
              result[0] += -0.017940895916117975;
            } else {
              result[0] += -0.0023189026352055772;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.0022161521511695494;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.033415387846675214;
              } else {
                result[0] += 0.011542628715227765;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0003431285874159346;
              } else {
                result[0] += -0.02629293642529387;
              }
            }
          } else {
            result[0] += 0.01077550571311898;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.002298429426695367;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
            result[0] += -0.01605766330074942;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
                result[0] += -0.004213155315488804;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.00014560495601593113;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                    result[0] += 0.0023825801265333654;
                  } else {
                    result[0] += 0.015666266989686397;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                      result[0] += 0.15691944067966623;
                    } else {
                      result[0] += -0.004272496607606239;
                    }
                  } else {
                    result[0] += 0.013033520090641501;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.04175038608922246;
                    } else {
                      result[0] += 0.041602233536523925;
                    }
                  } else {
                    result[0] += 0.01765350570212602;
                  }
                }
              } else {
                result[0] += 0.03692957554975309;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.013768056385050312;
        } else {
          result[0] += -0.019160940518991136;
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += -0.03105826053053272;
      } else {
        result[0] += -0.0033366281692676497;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.007286643613592405;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
              result[0] += -0.0123773388668442;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.012795133223297256;
              } else {
                result[0] += -0.0038982087417977654;
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                result[0] += -0.011948805814568527;
              } else {
                result[0] += 0.042752716628772953;
              }
            } else {
              result[0] += -0.005573682820744375;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
              result[0] += -0.033350334108076855;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                  result[0] += 0.03908709081961175;
                } else {
                  result[0] += -0.00304084274782104;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += 0.07428285035716038;
                } else {
                  result[0] += -0.03400538468769939;
                }
              }
            }
          } else {
            result[0] += 0.0012918835119391282;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.04749691243526811;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.027664281522718187;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.012708899457607767;
                          } else {
                            result[0] += -0.03261189299504818;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.039778576798334846;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.02140602598660742;
                    } else {
                      result[0] += 0.018145728708613428;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.014274333158288367;
                  } else {
                    result[0] += -0.042558173015699974;
                  }
                }
              } else {
                result[0] += -0.016600333618474745;
              }
            } else {
              result[0] += -0.03368384920703202;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += -0.040484103301955655;
                } else {
                  result[0] += -0.011613043263652574;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.009429887716834331;
                } else {
                  result[0] += -0.01856071564151573;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0007075778303626692;
                } else {
                  result[0] += 0.08196525475704279;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.029835622503772453;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.007663320374079744;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                result[0] += -0.00246348271371907;
                              } else {
                                result[0] += -0.03773503773707313;
                              }
                            } else {
                              result[0] += 0.013021763166427426;
                            }
                          } else {
                            result[0] += 0.013958027699012186;
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
                            result[0] += -0.03379147501320497;
                          } else {
                            result[0] += 0.17530924524691605;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.006239403687963719;
                        } else {
                          result[0] += 0.08275305507287307;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.010637001069883703;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                        result[0] += 0.00796990319972973;
                      } else {
                        result[0] += 0.09582835771305202;
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
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.0012659398608317093;
        } else {
          result[0] += -0.024581345063487105;
        }
      } else {
        result[0] += -0.023103164876683565;
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.11258693149912319;
        } else {
          result[0] += -0.024508988283074646;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.509355545043946201) ) ) {
            result[0] += 0.008777920186454227;
          } else {
            result[0] += -0.004329511036586405;
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
              result[0] += 0.010463715379704384;
            } else {
              result[0] += 0.08324218565101311;
            }
          } else {
            result[0] += -0.010564312891192956;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.056097030639650214) ) ) {
          result[0] += 0.03202428852463014;
        } else {
          result[0] += -0.046127462520523625;
        }
      } else {
        result[0] += -0.00018510051288245517;
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
            result[0] += 0.06692338516304325;
          } else {
            result[0] += -0.062437165162194865;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.0004484966895747321;
          } else {
            result[0] += -0.052927278297068975;
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.0044040965698276275;
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.034659259092147864;
              } else {
                result[0] += -0.008293811224995297;
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.018694345877473255;
                  } else {
                    result[0] += 0.01561680676697799;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.007874031102679816;
                  } else {
                    result[0] += 0.02355629596633113;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.031955918126331744;
                } else {
                  result[0] += 0.008287158901872843;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.023737025061309768;
                } else {
                  result[0] += 0.08939949376397481;
                }
              } else {
                result[0] += 0.014022348232256253;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.015714583793239312;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.02697941651234589;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.016559418737664852;
                    } else {
                      result[0] += 0.08261215615415235;
                    }
                  }
                }
              } else {
                result[0] += -0.032592207296262686;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.341600894927979404) ) ) {
                  result[0] += -0.022693284441660713;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                    result[0] += 0.04151400642336352;
                  } else {
                    result[0] += 0.14745801085914098;
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.08369975597233537;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                          result[0] += 0.03720175287107675;
                        } else {
                          result[0] += -0.09759172391956364;
                        }
                      } else {
                        result[0] += -0.02928162404065697;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.015831112231989124;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.062026315053137965;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                          result[0] += -0.0970045336927573;
                        } else {
                          result[0] += 0.011246510062418395;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.0377283699017425;
                }
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.006812767473583234;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                      result[0] += 0.020167764074391213;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.547126770019532138) ) ) {
                        result[0] += 0.036459694289449035;
                      } else {
                        result[0] += -0.144282325129758;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                      result[0] += -0.025722695142092057;
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.08409953861589198;
                      } else {
                        result[0] += 0.036054825853221055;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                  result[0] += 0.058863588157071524;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                    result[0] += -0.025726363413073973;
                  } else {
                    result[0] += 0.04689724426760496;
                  }
                }
              }
            }
          } else {
            result[0] += -0.00044582910270529784;
          }
        } else {
          result[0] += 0.010871236903573885;
        }
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.02349649070321113;
          } else {
            result[0] += -0.038711100660112636;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.255632162094117099) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                result[0] += -0.003084713348367006;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.744781017303467685) ) ) {
                  result[0] += 0.0018629947806411658;
                } else {
                  result[0] += -0.023538109054618922;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.007726618430623214;
              } else {
                result[0] += 0.008437561726617407;
              }
            }
          } else {
            result[0] += 0.004031666846031695;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.0017113883389959902;
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.05532312065205867;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.014140445812608773;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.09189967392920019;
                    } else {
                      result[0] += -0.04014859617969631;
                    }
                  } else {
                    result[0] += -0.008663514514368227;
                  }
                }
              }
            } else {
              result[0] += -0.004786189101674456;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
              result[0] += -0.0007835581825514905;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += -0.07924390671676058;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.019077402960725878;
                  } else {
                    result[0] += -0.023224773324703425;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.05878262085344859;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += 0.06327788678326143;
                        } else {
                          result[0] += -0.009343192634355929;
                        }
                      } else {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.03496725826275089;
                        } else {
                          result[0] += 0.005346477932365315;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.013619093370303557;
                    } else {
                      result[0] += 0.008539290626124988;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += 0.10715668501182687;
                } else {
                  result[0] += 0.03325922670654673;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12242221832275568) ) ) {
                  result[0] += 0.058171187881504244;
                } else {
                  result[0] += -0.06215644450807868;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                result[0] += 0.0704600544264152;
              } else {
                result[0] += -0.01587341853855545;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.025502115720962927;
                } else {
                  result[0] += -0.006144272772932606;
                }
              } else {
                result[0] += -0.004302796595986264;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.303166389465332919) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
                  result[0] += 0.002341574482822325;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.344550132751465732) ) ) {
                    result[0] += -0.03922113696489853;
                  } else {
                    result[0] += -0.01315452668265997;
                  }
                }
              } else {
                result[0] += 0.012411886087526856;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += -0.027434100640039927;
            } else {
              result[0] += 0.00394511982386379;
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    result[0] += -0.06914465892954655;
                  } else {
                    result[0] += 0.008661478041234048;
                  }
                } else {
                  result[0] += 0.032805132986194564;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                  result[0] += -0.026574749643874847;
                } else {
                  result[0] += 0.10503141461958182;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.287653446197511542) ) ) {
                result[0] += -0.02707361203122783;
              } else {
                result[0] += 0.08501989077121284;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
          result[0] += -0.008025751033775943;
        } else {
          result[0] += 0.0648706965471883;
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.418317794799805576) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0022811579706483935;
            } else {
              result[0] += -0.0947973915432957;
            }
          } else {
            result[0] += 0.044004718955541845;
          }
        } else {
          result[0] += 0.00011375420983141123;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += -0.02045779709504897;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.004357012876401204;
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.033439790374470875;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0018331136784530727;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.843275547027588779) ) ) {
                    result[0] += 0.0061609702687401915;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                      result[0] += -0.009107722858118677;
                    } else {
                      result[0] += -0.06562910458731375;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.007636970273962084;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.465643882751465732) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.046971228862344644;
                  } else {
                    result[0] += 0.011086651396353814;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += 0.10008389621276954;
                  } else {
                    result[0] += -0.08112808782683353;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.0026129451672102853;
                } else {
                  result[0] += 0.017765454044970992;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.01339028338932187;
              } else {
                result[0] += -0.037680854170669256;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.0002574312639293121;
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += 0.01794343166815721;
        } else {
          result[0] += -0.004587625013071856;
        }
      } else {
        result[0] += -0.02349695139319388;
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
              result[0] += 0.04773969755052959;
            } else {
              result[0] += -0.007354745002313978;
            }
          } else {
            result[0] += -0.018999321048121364;
          }
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.07981200613329453;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.047209475464269006;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.247576236724854404) ) ) {
                result[0] += 0.03361947478741751;
              } else {
                result[0] += 0.0035259310617921664;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                    result[0] += -0.15468261077086373;
                  } else {
                    result[0] += 0.000821291707003288;
                  }
                } else {
                  result[0] += -0.036892839841077184;
                }
              } else {
                result[0] += 0.00943032010147183;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.005386529936870694;
                } else {
                  result[0] += -0.022670059394910947;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.0017668933023443504;
                  } else {
                    result[0] += -0.03133456817757368;
                  }
                } else {
                  result[0] += 0.0006552673137872674;
                }
              }
            }
          } else {
            result[0] += 0.00654315919348309;
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.0026996301681136456;
            } else {
              result[0] += -0.0157050948868493;
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.865389823913576) ) ) {
                result[0] += 0.0372554096558167;
              } else {
                result[0] += -0.0689670220624028;
              }
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                  result[0] += 0.007537031563695274;
                } else {
                  result[0] += 0.02147363262602687;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                  result[0] += -0.021086194531765728;
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += -0.013733674400424573;
                    } else {
                      result[0] += -0.07715265196077031;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += -0.01693296089202377;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.0391480294995086;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.353313446044923651) ) ) {
                            result[0] += -0.11463375773121712;
                          } else {
                            result[0] += 0.01813592441090943;
                          }
                        } else {
                          result[0] += -0.15331406731315084;
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
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.017797946929933417) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.0012969247226025508;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
            result[0] += -0.007930982716298137;
          } else {
            result[0] += -0.0517774368225047;
          }
        } else {
          result[0] += 0.00994135437440192;
        }
      }
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
          result[0] += -0.01433723189682801;
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.921060562133789951) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.0016474004207925601;
              } else {
                result[0] += -0.01687856603203774;
              }
            } else {
              result[0] += -0.012352109335164348;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.010841972329744015;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.08745531248072858;
                        } else {
                          result[0] += 0.004503731668122091;
                        }
                      } else {
                        result[0] += -0.058819182098463346;
                      }
                    } else {
                      result[0] += 0.032211233651870654;
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.024221492973558217;
                    } else {
                      result[0] += -0.014256380876225595;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.002185447933565924;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.050162147435669796;
                      } else {
                        result[0] += -0.04349910218801789;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.049070074815409696;
                      } else {
                        result[0] += 0.0027352883699478157;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                        result[0] += -0.06988553999233492;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                            result[0] += -0.016897955536948644;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                              result[0] += 0.002473691944173856;
                            } else {
                              result[0] += 0.07132930678505119;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += 0.008727522531603442;
                          } else {
                            result[0] += 0.07082688777896586;
                          }
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
                result[0] += -0.024469793192266637;
              } else {
                result[0] += 0.0009295475239050224;
              }
            }
          }
        }
      } else {
        result[0] += -0.038332763104960296;
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.003075796661801411;
        } else {
          result[0] += -0.02247705457287459;
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.006801741343428033;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    result[0] += -0.04804545865580177;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.003940576599039599;
                    } else {
                      result[0] += -0.04248142373969611;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
                    result[0] += 0.008600927272806275;
                  } else {
                    result[0] += 0.1828300701630794;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.00261302815306315;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                    result[0] += 0.06170039249480852;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.023546784794846596;
                    } else {
                      result[0] += -0.05779173888745223;
                    }
                  }
                }
              } else {
                result[0] += -0.07437446520547805;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += -0.0654464017133097;
              } else {
                result[0] += 0.00946772870860091;
              }
            } else {
              result[0] += 0.02762860242865008;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.01305690861686752;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205624103546144354) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.136462926864624912) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                    result[0] += 0.007468793085844504;
                  } else {
                    result[0] += -0.1764390204052205;
                  }
                } else {
                  result[0] += -0.033907325386981435;
                }
              } else {
                result[0] += -0.01787878869566659;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.056564234947033804;
                } else {
                  result[0] += -0.0002767593483148703;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.088880300521851474) ) ) {
                  result[0] += -0.010738319459651945;
                } else {
                  result[0] += -0.03298048123682811;
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.008324666972330398;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.119004011154175693) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.863673448562622958) ) ) {
                        result[0] += -0.0021745553014376833;
                      } else {
                        result[0] += 0.005868411651229821;
                      }
                    } else {
                      result[0] += 0.007752737203445177;
                    }
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                        result[0] += -0.0021184883890096916;
                      } else {
                        result[0] += -0.06376960663723102;
                      }
                    } else {
                      result[0] += 0.012310888709578754;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.026955452580249076;
                } else {
                  result[0] += -0.014836982628177725;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
            result[0] += 0.015390707853318959;
          } else {
            result[0] += -0.009359659779356247;
          }
        } else {
          result[0] += -0.03521869412813335;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.535966873168947089) ) ) {
          result[0] += 0.0007392625674346604;
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += -0.03251535929909068;
                } else {
                  result[0] += -0.0007167967565011586;
                }
              } else {
                result[0] += 0.05748468857822684;
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.038058340929713;
                    } else {
                      result[0] += 0.04609096638772936;
                    }
                  } else {
                    result[0] += -0.010959282058389946;
                  }
                } else {
                  result[0] += -0.10204774783600437;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                      result[0] += -0.03967825712744971;
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.02367195287482109;
                      } else {
                        result[0] += 0.02189445635441513;
                      }
                    }
                  } else {
                    result[0] += -0.0482293888860377;
                  }
                } else {
                  result[0] += 0.030110131507206678;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.002396207293115787;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.031375127853345425;
                } else {
                  result[0] += -0.005468905656162034;
                }
              } else {
                result[0] += 0.017570351697608216;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.285748958587648261) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
              result[0] += 0.023100626173917532;
            } else {
              result[0] += -0.030867524914011842;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += 0.05475033083099219;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.015291282569362245;
                } else {
                  result[0] += 0.12252678167758496;
                }
              } else {
                result[0] += -0.04523075533517025;
              }
            }
          }
        } else {
          result[0] += -0.09584584406359578;
        }
      } else {
        result[0] += -0.05111821609859927;
      }
    } else {
      result[0] += 0.1643356611847549;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.0007294506503600123;
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.872538805007935458) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += -0.04121003611627475;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                    result[0] += 5.9109776032519525e-06;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                        result[0] += -0.02685416077882415;
                      } else {
                        result[0] += 0.03540211529013757;
                      }
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.02132966015602099;
                      } else {
                        result[0] += -0.010069198680396066;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.005798736121743129;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.0634273820366855;
                    } else {
                      result[0] += -0.002384640084359967;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.03276979344737699;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += -0.0894833462341499;
                    } else {
                      result[0] += 0.0022623075803127482;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.06175391877433583;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.623839378356934482) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                  result[0] += -0.17062542588101656;
                } else {
                  result[0] += -0.002333101178250211;
                }
              } else {
                result[0] += 0.006085341660063211;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                result[0] += 0.0043767167101065275;
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.011229126708448132;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.03242839922720728;
                    } else {
                      result[0] += 0.07953481138692742;
                    }
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.11323336472326391;
                    } else {
                      result[0] += 0.0008372665156835954;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.270308971405030185) ) ) {
            result[0] += -0.013047955024434075;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.005634784184346392;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                result[0] += 0.004647303355615726;
              } else {
                result[0] += 0.07045550401008749;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.868834793567657693) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += 0.17180738674377513;
              } else {
                result[0] += -0.007007286914157787;
              }
            } else {
              result[0] += -0.005004021420446163;
            }
          } else {
            result[0] += 0.026618864144038953;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.198252916336060458) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
              result[0] += 0.20473575745733597;
            } else {
              result[0] += -0.008390091818883286;
            }
          } else {
            result[0] += -0.03886500122937864;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.014179931184770995;
          } else {
            result[0] += 0.0036229013435789565;
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.027022402685846714;
          } else {
            result[0] += 0.0018773236460252907;
          }
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += -0.022972333858864404;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.008696547326365467;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.06422556231142464;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.020740199974932125;
                  } else {
                    result[0] += 0.04767118985388594;
                  }
                } else {
                  result[0] += -0.036756719465466735;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.013209877251769461;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                    result[0] += 0.08003744621687935;
                  } else {
                    result[0] += -0.1355660414596207;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
                      result[0] += 0.015547154204828766;
                    } else {
                      result[0] += -0.04916880281396202;
                    }
                  } else {
                    result[0] += 0.03541228055748819;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01338488041048832;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.011351387789713483;
                        } else {
                          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.0777259850032291;
                          } else {
                            result[0] += -0.030574514742294213;
                          }
                        }
                      } else {
                        result[0] += 0.016042971891723108;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                          result[0] += -0.009058394489772846;
                        } else {
                          result[0] += -0.0987468479330368;
                        }
                      } else {
                        result[0] += 0.03560841644846998;
                      }
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                          result[0] += 0.059634460759632746;
                        } else {
                          result[0] += 0.02132258318881175;
                        }
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
                          result[0] += -0.007999277421407767;
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.021593931627858828;
                          } else {
                            result[0] += -0.011483971283078934;
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
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
          result[0] += 0.000970426052921854;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
            result[0] += -0.0009757529772140031;
          } else {
            result[0] += 0.042181340535827984;
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.960767745971680132) ) ) {
          result[0] += -0.0022024382788160476;
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.020784084413669635;
          } else {
            result[0] += -0.005250863090501902;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          result[0] += -0.0008818424742881551;
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.069797992706300604) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.215607166290284091) ) ) {
                  result[0] += -0.012191669477159994;
                } else {
                  result[0] += -0.15755653037386305;
                }
              } else {
                result[0] += 0.04229169542086959;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.13810903851929385;
                } else {
                  result[0] += 0.016402080332910584;
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.019330318672976042;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.198464870452881303) ) ) {
                    result[0] += -0.044481004622793203;
                  } else {
                    result[0] += -0.003507243832601348;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.06083172167814757;
              } else {
                result[0] += 0.016408814756253685;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.014271669808455512;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.07735933465097271;
                    } else {
                      result[0] += -0.09254667316169121;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.733257532119751865) ) ) {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.03391643171416802;
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.453179836273194248) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                                result[0] += -0.13223987055785413;
                              } else {
                                result[0] += -0.07142247276884688;
                              }
                            } else {
                              result[0] += 0.08473058386984261;
                            }
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                              result[0] += -0.08970208719270312;
                            } else {
                              result[0] += -0.026510288954632467;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.02457705886165551;
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.088880300521851474) ) ) {
                          result[0] += 0.04671982569396623;
                        } else {
                          result[0] += -0.051592158148447004;
                        }
                      } else {
                        result[0] += 0.04847886847537008;
                      }
                    }
                  }
                } else {
                  result[0] += 0.0015691188045080882;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += 0.006955216081666588;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.0897429844653353;
                } else {
                  result[0] += -0.01592521473662108;
                }
              } else {
                result[0] += -0.00048437616684043894;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.01661899139022785;
              } else {
                result[0] += -0.02085796142574286;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                  result[0] += -0.0006623815508368777;
                } else {
                  result[0] += 0.010940617994598228;
                }
              } else {
                result[0] += 0.03764745469777035;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.14101039231911291;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.698768615722657138) ) ) {
                  result[0] += 0.024896218675312255;
                } else {
                  result[0] += -0.08792571725069438;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                      result[0] += 0.03204591885145986;
                    } else {
                      result[0] += -0.09511635275548125;
                    }
                  } else {
                    result[0] += -0.07004654434671011;
                  }
                } else {
                  result[0] += -0.08822487809792864;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                    result[0] += 0.0034498523623266936;
                  } else {
                    result[0] += 0.042701284038711484;
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.043143215464387444;
                  } else {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.006533475499316845;
                    } else {
                      result[0] += -0.021501032795588227;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += 0.015097387846615701;
                  } else {
                    result[0] += -0.026011557163431998;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                    result[0] += 0.009972303013041105;
                  } else {
                    result[0] += 0.029901814446615046;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        result[0] += 0.02583326538069496;
      } else {
        result[0] += 0.0014088053130342803;
      }
    } else {
      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
        result[0] += 0.00890556264909982;
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.06621293256951934;
          } else {
            result[0] += 0.0016516630747258826;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.01266221474194771;
          } else {
            result[0] += -0.05343732841056729;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        result[0] += -0.12613341563352753;
      } else {
        result[0] += 0.0038402610501204674;
      }
    } else {
      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
                result[0] += -0.022467660500392193;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                  result[0] += 0.06956493088020532;
                } else {
                  result[0] += 0.006240772140805751;
                }
              }
            } else {
              result[0] += 0.008227965453314614;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.181854770270237;
              } else {
                result[0] += 0.0016594193962541698;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7.592854261398316318) ) ) {
                    result[0] += -0.053274837994132886;
                  } else {
                    result[0] += 0.1479516644366308;
                  }
                } else {
                  result[0] += -0.01798257148441692;
                }
              } else {
                result[0] += -0.0038568144883940354;
              }
            }
          }
        } else {
          result[0] += -0.024081878411252658;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.1391338201574793;
                  } else {
                    result[0] += 0.038335227444164476;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.83939445018768355) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                      result[0] += -0.1329322556914041;
                    } else {
                      result[0] += 0.009606262777576166;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0194392746091014;
                      } else {
                        result[0] += -0.011299472470398243;
                      }
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.505036115646363193) ) ) {
                        result[0] += -0.04744123218765879;
                      } else {
                        result[0] += -0.0022249581567937034;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += 0.004021623716103561;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.007415420725139114;
                      } else {
                        result[0] += -0.03210810714678943;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.04679774200714793;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
                            result[0] += -0.08728947405555303;
                          } else {
                            result[0] += 0.005513686598033602;
                          }
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.08649364460404468;
                            } else {
                              result[0] += 0.0014504494743827682;
                            }
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.020181510273252035;
                            } else {
                              result[0] += -0.02773177249484427;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.02514482230714181;
                  } else {
                    result[0] += 0.00034247672426322164;
                  }
                }
              }
            } else {
              result[0] += 0.05744968757834634;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.07638253734019573;
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.893023490905762607) ) ) {
                result[0] += 0.0014868438393173288;
              } else {
                result[0] += -0.003322205248962345;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.242078304290772373) ) ) {
                  result[0] += 0.006547647131188901;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += 0.05837229799054649;
                    } else {
                      result[0] += -0.036665029154325776;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                      result[0] += 0.01220779723040798;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.018181017871545187;
                        } else {
                          result[0] += -0.04650502744209419;
                        }
                      } else {
                        result[0] += 0.00294768112358612;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.909855604171753818) ) ) {
                    result[0] += 0.001000518682409377;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.664408206939698154) ) ) {
                        result[0] += 0.006306735330486184;
                      } else {
                        result[0] += -0.027694533989562165;
                      }
                    } else {
                      result[0] += -0.0017028702018425414;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    result[0] += -0.005279289487203313;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.03674152056064488;
                      } else {
                        result[0] += -0.023384446574018364;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.35441589355468928) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.341600894927979404) ) ) {
                            result[0] += -0.010526885072404131;
                          } else {
                            result[0] += 0.012031331096435754;
                          }
                        } else {
                          result[0] += 0.024945795393737036;
                        }
                      } else {
                        result[0] += 0.04980155801823661;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += -0.013890367292324193;
              } else {
                result[0] += 0.03934910144135029;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.011176109964093066;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.18054183668902465;
              } else {
                result[0] += 0.00745002859719001;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
      result[0] += 0.0011898504959731821;
    } else {
      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
        result[0] += -0.08615870687283177;
      } else {
        result[0] += -0.01250991654752589;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.12450933456421076) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.002590865704284012;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += -0.011835547793162638;
            } else {
              result[0] += -0.05243136576173927;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
              result[0] += 0.0011854472900230469;
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                    result[0] += 0.003957177342664685;
                  } else {
                    result[0] += -0.06195982542778514;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.847591876983644354) ) ) {
                    result[0] += -0.04389586915907817;
                  } else {
                    result[0] += 0.025049825349865048;
                  }
                }
              } else {
                result[0] += -0.09385158742902061;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.006656338539983778;
              } else {
                result[0] += 0.015692421302813685;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.029846876755043297;
                } else {
                  result[0] += 0.012659476364606305;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.280659198760987216) ) ) {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.07025162473743583;
                  } else {
                    result[0] += 0.02724059932796591;
                  }
                } else {
                  result[0] += -0.03092193331839396;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.011704324106924981;
      }
    } else {
      result[0] += 0.011434759974187417;
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.48738741874694913) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.024349402767208875;
              } else {
                result[0] += -0.02419127861381315;
              }
            } else {
              result[0] += 0.08497653502036515;
            }
          } else {
            result[0] += -0.019568820919623336;
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                  result[0] += -0.013939159831322543;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.08930264534102822;
                    } else {
                      result[0] += -0.0515233223036729;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                      result[0] += 0.11379472541881858;
                    } else {
                      result[0] += -0.01909473649729355;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.006730998190293468;
                } else {
                  result[0] += 0.04455944333809563;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.021864807448872495;
                } else {
                  result[0] += -0.0245587537333284;
                }
              } else {
                result[0] += -0.06104814573403421;
              }
            }
          } else {
            result[0] += -0.02832446718510608;
          }
        }
      } else {
        result[0] += -0.05007792370947224;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.967588424682618964) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.14335426576719879;
            } else {
              result[0] += 0.00027261841389243435;
            }
          } else {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += -0.011759208903811666;
              } else {
                result[0] += 0.017406815313233123;
              }
            } else {
              result[0] += -0.008977907150119922;
            }
          }
        } else {
          result[0] += -0.015025564296734016;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.011088429917909398;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += -0.03221340978399763;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.008377121780787148;
                    } else {
                      result[0] += 0.08294655953223223;
                    }
                  }
                }
              } else {
                result[0] += -0.059039872160406064;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.018767554384428328;
              } else {
                result[0] += 0.011576046997532414;
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.01957245192264105;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.03175123555055427;
                    } else {
                      result[0] += 0.011862192501878168;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.49584054946899592) ) ) {
                        result[0] += 0.009627328214471828;
                      } else {
                        result[0] += -0.025707468624181936;
                      }
                    } else {
                      result[0] += -0.03559110805719498;
                    }
                  }
                } else {
                  result[0] += 0.007854931108401476;
                }
              }
            } else {
              result[0] += 0.0061908642214158534;
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.026570871745283744;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += 0.02329026705025799;
                } else {
                  result[0] += -0.02789237793784033;
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.004224016352824418;
                } else {
                  result[0] += 0.030138234115919885;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.006991621372726842;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.026837247291323425;
              } else {
                result[0] += -0.05930677237076707;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.12450933456421076) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.69067406654357999) ) ) {
            result[0] += 0.01760521668317434;
          } else {
            result[0] += -0.01548194738739752;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
              result[0] += 0.0012975925234710605;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.14301252365112482) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.022773639416288478;
                    } else {
                      result[0] += -0.026088114334082882;
                    }
                  } else {
                    result[0] += -0.022658454978931087;
                  }
                } else {
                  result[0] += -0.008877686015223187;
                }
              } else {
                result[0] += 0.09779358023492608;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                result[0] += 0.0003542891572834395;
              } else {
                result[0] += -0.014517516232717432;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.052919648650468123;
                  } else {
                    result[0] += 0.010195390084352201;
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.014047937747872342;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.036704712673380664;
                    } else {
                      result[0] += 0.0010390337158895833;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.014214379060340288;
                    } else {
                      result[0] += -0.011157046345050683;
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.011395276287308546;
                    } else {
                      result[0] += -0.02952390671850851;
                    }
                  }
                } else {
                  result[0] += 0.019876288035891945;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.010462309274651312;
      }
    } else {
      result[0] += 0.010831258017231695;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
      result[0] += -0.010326390259348834;
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.384830474853516513) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.025732332679953576;
                } else {
                  result[0] += 0.10083770307212914;
                }
              } else {
                result[0] += -0.015520403358497829;
              }
            } else {
              result[0] += -0.016445734651852825;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.016144440668774195;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += 0.001039641231366908;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.0844370117646675;
                      } else {
                        result[0] += 0.020899632638324745;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                      result[0] += -0.006538310282331542;
                    } else {
                      result[0] += 0.10953004593139715;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                  result[0] += 0.020612628460418673;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.019868962566710814;
                  } else {
                    result[0] += -0.060612202746705736;
                  }
                }
              }
            } else {
              result[0] += -0.025096444070048924;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
              result[0] += 0.0025211170848528784;
            } else {
              result[0] += -0.0469390099567884;
            }
          } else {
            result[0] += 0.17691563369926583;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.553712725639343706) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.680161952972413886) ) ) {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.013910713196653638;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                    result[0] += 0.001374254356190085;
                  } else {
                    result[0] += 0.06554676191767475;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.016049893787928017;
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.02500629370298913;
                  } else {
                    result[0] += -0.04151014790700597;
                  }
                }
              }
            } else {
              result[0] += 0.04262859150633519;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0008607782684229731;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.017173836426663314;
                  } else {
                    result[0] += -0.01098715366422908;
                  }
                }
              } else {
                result[0] += -0.016826963899325295;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.010355941416693671;
                  } else {
                    result[0] += 0.0021498512247594415;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.010455126263280136;
                    } else {
                      result[0] += 0.007065456512974605;
                    }
                  } else {
                    result[0] += 0.013520130284316218;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.012625858159334594;
                  } else {
                    result[0] += -0.0377107338768878;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
                    result[0] += 0.008919901445877077;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.030272217610692365;
                      } else {
                        result[0] += -0.008953442084709008;
                      }
                    } else {
                      result[0] += -0.0031001831000861974;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.02246022457556683;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
      result[0] += 0.0058428740244321;
    } else {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.06038897061475961;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.09894168532312608;
            } else {
              result[0] += -0.0017247368191336664;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.0016776442978815769;
                } else {
                  result[0] += -0.05424511624533793;
                }
              } else {
                result[0] += -0.04034365577781571;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.007911972582127743;
              } else {
                result[0] += 0.00227172303297094;
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.0006829237272739925;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                    result[0] += -0.004766088562296996;
                  } else {
                    result[0] += -0.033716471070075964;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0183203006086651;
                    } else {
                      result[0] += -0.14377874301463478;
                    }
                  } else {
                    result[0] += -0.052008735845693925;
                  }
                }
              } else {
                result[0] += -0.06743578807680604;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.363078355789185458) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.008337819613377899;
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              result[0] += 0.0008545696207124719;
            } else {
              result[0] += 0.010578343200460322;
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0012366991826748622;
                } else {
                  result[0] += 0.041397654929498366;
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.923617362976075107) ) ) {
                    result[0] += -0.003420762503667413;
                  } else {
                    result[0] += -0.03869358354485247;
                  }
                } else {
                  result[0] += 0.004680660401700487;
                }
              }
            } else {
              result[0] += 0.02832860604143276;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.12803614943725547;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.07661493169155627;
                    } else {
                      result[0] += 0.016006143703711596;
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.14504551944540242;
                    } else {
                      result[0] += 0.04010087885821409;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.688684463500978339) ) ) {
                  result[0] += -0.0038600885220799967;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                    result[0] += -0.07097801890776907;
                  } else {
                    result[0] += 0.027209568331355134;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.014044377465054186;
                } else {
                  result[0] += -0.09450632668467386;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.002049135938897905;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.007378136908436393;
                  } else {
                    result[0] += 0.0644447538488187;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
      result[0] += -0.010073118410536224;
    } else {
      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.012868134143070191;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.07097845342143858;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                    result[0] += 0.03600857044740643;
                  } else {
                    result[0] += -0.05989293483055749;
                  }
                } else {
                  result[0] += -0.0153342470010296;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
              result[0] += 0.029736320468871598;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.040802897884938495;
                  } else {
                    result[0] += 0.010982651816195765;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.0027147702384669537;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.0691950066577679;
                    } else {
                      result[0] += -0.0024709166054088572;
                    }
                  }
                }
              } else {
                result[0] += 0.0247467441636384;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.010292254365898137;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    result[0] += 0.0029591796520662757;
                  } else {
                    result[0] += -0.013393124690418168;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.83939445018768355) ) ) {
                    result[0] += 0.018760010082712324;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += 0.04042804300213081;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.579273939132691318) ) ) {
                        result[0] += -0.040995556604831535;
                      } else {
                        result[0] += -0.014431289844611648;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0009058998088571583;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.007754600805833115;
            } else {
              result[0] += -0.040371076554054634;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.005149513042603894;
        } else {
          result[0] += -0.0328046668379826;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.013717691044085524;
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.01643112837269974;
              } else {
                result[0] += 0.00166478223483853;
              }
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.01547987367768128;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                  result[0] += -0.00367614014123984;
                } else {
                  result[0] += 0.04536091829909577;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.038891286749238196;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                  result[0] += -0.034264249117276206;
                } else {
                  result[0] += -1.5866537076331444e-05;
                }
              }
            } else {
              result[0] += -0.03560657811300982;
            }
          } else {
            result[0] += 0.003273755489507136;
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.007776952241109594;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0390576123704273;
              } else {
                result[0] += -0.08131519119088206;
              }
            } else {
              result[0] += -0.00906269689560987;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.011657900281846833;
          } else {
            result[0] += 0.021336047714893218;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0025657167195329306;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.009428888871839305;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.032540758117047035;
              } else {
                result[0] += -0.0572418487417276;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += 0.033721787787050796;
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.00613668021506277;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                result[0] += 0.09848653601274086;
              } else {
                result[0] += -0.03454182891543246;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.007011976446964555;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.006383974601828594;
          } else {
            result[0] += 0.001923700534617343;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18134641647339045) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.007954690334875681;
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.024840562016309638;
                } else {
                  result[0] += -0.01065653181105416;
                }
              } else {
                result[0] += 0.0055109563578240395;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += 0.13351123473598867;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.48738741874694913) ) ) {
                      if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.01080762071131719;
                      } else {
                        result[0] += 0.0195544751140584;
                      }
                    } else {
                      result[0] += 0.03815074657897222;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.008325592404744125;
                    } else {
                      result[0] += 0.027756340093306294;
                    }
                  } else {
                    result[0] += -0.032804419938097254;
                  }
                }
              } else {
                result[0] += 0.03780757722241716;
              }
            }
          } else {
            result[0] += -0.013259938027420702;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.277936458587647373) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.02793322301109484;
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
              result[0] += 0.0010964740928770644;
            } else {
              result[0] += -0.036451968090390384;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.015622751161210206;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.017579922838804808;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                      result[0] += -0.014682633015560862;
                    } else {
                      result[0] += 0.03803608272254094;
                    }
                  } else {
                    result[0] += 0.023568835884134814;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.05727226324768662;
                  } else {
                    result[0] += 0.06797542171482719;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.795494556427002841) ) ) {
                      result[0] += -0.032024219788535035;
                    } else {
                      result[0] += 0.14811455591086858;
                    }
                  } else {
                    if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.909254074096680576) ) ) {
                        result[0] += 0.011889785459476778;
                      } else {
                        result[0] += 0.13279261110747373;
                      }
                    } else {
                      result[0] += -0.05546676808395782;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
              result[0] += -0.039808553885026306;
            } else {
              result[0] += 0.23207808221642823;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59476566314697443) ) ) {
            result[0] += -0.01105866148622027;
          } else {
            result[0] += -0.030372594123169123;
          }
        } else {
          result[0] += -0.07786128202452425;
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.005243693453826286;
        } else {
          result[0] += -0.05141723794683675;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    result[0] += 0.0010171016806578362;
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
      result[0] += -0.01363926958979381;
    } else {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.020848005827403305;
          } else {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.016100111404642497;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                result[0] += -0.019911587279210767;
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.023028693920109547;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                    result[0] += 0.04601434310057249;
                  } else {
                    result[0] += -0.035651427223153204;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.0700730996999845;
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += 0.05059671797023721;
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.01683842132319786;
                } else {
                  result[0] += -0.05055042306746454;
                }
              }
            } else {
              result[0] += 0.01115306434391919;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.02541238330063716;
              } else {
                result[0] += 0.047041679443044375;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                          result[0] += -0.01377947044657265;
                        } else {
                          result[0] += 0.06709182982157644;
                        }
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.255004644393921787) ) ) {
                            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                                result[0] += -0.012695778876919953;
                              } else {
                                result[0] += 0.01742018712480552;
                              }
                            } else {
                              result[0] += 0.033883734884089134;
                            }
                          } else {
                            result[0] += -0.04228687002878253;
                          }
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.037814961414298895;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += 0.03014877002757653;
                            } else {
                              result[0] += -0.0031851215560136955;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                        result[0] += 0.006310163328912745;
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.04267792885421426;
                        } else {
                          result[0] += -0.004543147736496402;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.02562200799700296;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.003850105570044763;
                        } else {
                          result[0] += -0.0479337671804705;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.42478513717651456) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.017326331681059626;
                        } else {
                          result[0] += 0.027928088887441;
                        }
                      } else {
                        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.033802021992490355;
                        } else {
                          result[0] += 0.09238458875451963;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.04224331967025666;
                }
              } else {
                result[0] += -0.03300041125492921;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.036049604415894443) ) ) {
                  result[0] += 0.008279471856865506;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                    result[0] += 0.019942683493900812;
                  } else {
                    result[0] += -0.02036492505819558;
                  }
                }
              } else {
                result[0] += -0.027084092114346703;
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                  result[0] += -0.03369196411860093;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.04594999894101762;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.007335473790617482;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.190353393554689276) ) ) {
                        result[0] += -0.0013733445810292597;
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.04508387372565542;
                        } else {
                          result[0] += -0.0013438382524026777;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += 0.06434324779482585;
                    } else {
                      result[0] += -0.0061859603393471865;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                          result[0] += 0.057613623583297316;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.97070193290710538) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                              result[0] += 0.08347449782490513;
                            } else {
                              result[0] += -0.041555119969001804;
                            }
                          } else {
                            result[0] += -0.00867168391625665;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                                result[0] += -0.023936216141141782;
                              } else {
                                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                                  result[0] += 0.012728127575418741;
                                } else {
                                  result[0] += -0.07203433787642233;
                                }
                              }
                            } else {
                              result[0] += -0.057642222694262404;
                            }
                          } else {
                            result[0] += 0.02416386588745895;
                          }
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.749434947967529741) ) ) {
                            result[0] += -0.017571977897256872;
                          } else {
                            result[0] += -0.05388192782926522;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.057565962638026186;
                    }
                  }
                } else {
                  result[0] += 0.03269155837312547;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.00010835596682034316;
            } else {
              result[0] += 0.009372063613679659;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.007271400912530076;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.0019964204587520937;
          } else {
            result[0] += -0.025558941417803866;
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.006718651615818726;
              } else {
                result[0] += -0.05444913780142953;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.017871992765742098;
              } else {
                result[0] += -0.0555769925884427;
              }
            }
          } else {
            result[0] += 0.01094180086656573;
          }
        } else {
          result[0] += -0.036992266279386586;
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += 0.0034335870809988306;
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.040802041646689706;
          } else {
            result[0] += -0.0034818037511468365;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.009607856918941862;
          } else {
            result[0] += 0.0024345882802204837;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18134641647339045) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.579273939132691318) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
          result[0] += -0.007260550922580543;
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0022538281385542746;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.033982265660404264;
                  } else {
                    result[0] += -0.004739702841134866;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.0207848214019105;
                    } else {
                      result[0] += -0.03279392996018082;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.848652839660646308) ) ) {
                      result[0] += 0.09198290866289426;
                    } else {
                      result[0] += -0.027460518966467014;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.004105359156318275;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                        result[0] += -0.020898719322726814;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                            result[0] += 0.07706238210747285;
                          } else {
                            result[0] += 0.02505889189151564;
                          }
                        } else {
                          result[0] += 0.007941023203717848;
                        }
                      }
                    } else {
                      result[0] += 0.035373137599027854;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += -0.0022230512441537144;
                    } else {
                      result[0] += 0.024509147923306305;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                      result[0] += 0.020958188669815477;
                    } else {
                      result[0] += -0.016132233179094495;
                    }
                  }
                } else {
                  result[0] += 0.02396486155515371;
                }
              } else {
                result[0] += 0.03672990868132749;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
              result[0] += -0.003213755086998867;
            } else {
              result[0] += -0.03776265826244367;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.277936458587647373) ) ) {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.024742835026488493;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
              result[0] += -0.0007584702997890757;
            } else {
              result[0] += 0.01112968159234048;
            }
          }
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.014969941887446148;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.01615505525810341;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.016180796181266426;
                    } else {
                      result[0] += 0.015591946411109801;
                    }
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.0002636450412948398;
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.749434947967529741) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                          result[0] += 0.028055185971805476;
                        } else {
                          result[0] += -0.039556691032715134;
                        }
                      } else {
                        result[0] += 0.05582257067122353;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.030261463337217645;
                  } else {
                    result[0] += 0.06913768966127685;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.027630156798226724;
                  } else {
                    if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
                        result[0] += 0.011234156704877159;
                      } else {
                        result[0] += 0.11804811577217487;
                      }
                    } else {
                      result[0] += -0.04879072658133842;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.03792789829073054;
            } else {
              result[0] += 0.20416056260227688;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        result[0] += -0.018641292204956764;
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.004755939732999756;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.0001643685834582648;
              } else {
                result[0] += -0.051852025613103175;
              }
            }
          } else {
            if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.0015880914410138173;
            } else {
              result[0] += 0.02730763081823491;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.010599884773293653;
          } else {
            result[0] += -0.032228122205799806;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
          result[0] += 0.0004470450037997103;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += 0.0031592971812183924;
            } else {
              result[0] += -0.0328463953541806;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.113908529281617099) ) ) {
                result[0] += -0.037371267947784954;
              } else {
                result[0] += -0.004067586053213988;
              }
            } else {
              result[0] += 0.010570874108177366;
            }
          }
        }
      } else {
        result[0] += 0.0012127325682106571;
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.909254074096680576) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.806276082992554599) ) ) {
              result[0] += -0.007320028086959443;
            } else {
              result[0] += 0.03979117966615915;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.04485986970907371;
            } else {
              result[0] += 0.009308538281755022;
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                result[0] += -0.03230745975095541;
              } else {
                result[0] += 0.0048166005760707795;
              }
            } else {
              result[0] += 0.021445194064949633;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.050345611886619714;
              } else {
                result[0] += -0.007138551875794055;
              }
            } else {
              result[0] += 0.04026218750104957;
            }
          }
        }
      } else {
        result[0] += -0.09309956552604716;
      }
    }
  } else {
    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.022493233580307286;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.026195002511026102;
              } else {
                result[0] += -0.15482507398217774;
              }
            }
          } else {
            result[0] += 0.01880083871740592;
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.795426130294800249) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.0068277332087255845;
            } else {
              result[0] += -0.03532242308744609;
            }
          } else {
            result[0] += -0.03245651191068826;
          }
        }
      } else {
        result[0] += -0.03822952132338659;
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += -0.02896675438463496;
            } else {
              result[0] += 0.0009323676010702076;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.001520350837768225;
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.024165475282521578;
              } else {
                result[0] += 0.08632525542262284;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.012067908668950068;
                } else {
                  result[0] += 0.012738065416476072;
                }
              } else {
                result[0] += 0.04227449707202497;
              }
            } else {
              result[0] += -0.0209801068424196;
            }
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.0962251245474196;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
                        result[0] += -0.00231227499928542;
                      } else {
                        result[0] += -0.04278928782899319;
                      }
                    }
                  } else {
                    result[0] += -0.02811763696856442;
                  }
                } else {
                  result[0] += -0.005494025296443102;
                }
              } else {
                result[0] += -0.04936582191269039;
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += -0.026885402385997653;
                } else {
                  result[0] += 0.005452993397399817;
                }
              } else {
                result[0] += 0.03961004328716144;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.031045415890553818;
                } else {
                  result[0] += -0.05528211977490588;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.07482688427286968;
                    } else {
                      result[0] += -0.01302420353080879;
                    }
                  } else {
                    result[0] += -0.08395931045366278;
                  }
                } else {
                  result[0] += -0.14142044201321538;
                }
              }
            } else {
              result[0] += 0.007383035726139547;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                result[0] += 0.02810849709819623;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                  result[0] += 0.013041240203942534;
                } else {
                  result[0] += -0.01761197859477617;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02007972410722858;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.03032531065101772;
                  } else {
                    result[0] += 0.0728846283501439;
                  }
                }
              } else {
                result[0] += 0.07442852809308484;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.008981282245220162;
              } else {
                result[0] += 0.05826796063526035;
              }
            } else {
              result[0] += -0.039849442541441085;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.007220763808811541;
            } else {
              result[0] += -0.037184709722954;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY(  (data[37].missing != -1) && (data[37].fvalue <= (double)-1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
      result[0] += -0.1465926553101745;
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
        result[0] += 0.0066739562397509005;
      } else {
        result[0] += -0.0018670837120718293;
      }
    }
  } else {
    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
            result[0] += 0.012704413628066706;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.006572145498391473;
            } else {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                    result[0] += -0.14960436729042043;
                  } else {
                    result[0] += -0.02533367304055298;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                    result[0] += -0.04209343234554494;
                  } else {
                    result[0] += 0.020534602214042547;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
                  result[0] += -0.029017666072530097;
                } else {
                  result[0] += -0.08591588906282632;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.944020271301270419) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.006662012667769323;
              } else {
                result[0] += -0.033794731386405076;
              }
            } else {
              result[0] += 0.019235820664709885;
            }
          } else {
            result[0] += -0.030295253809212216;
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.02562952046607996;
            } else {
              result[0] += 0.0014495835037121017;
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.06546511718879794;
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += -0.05493726624589277;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.03880980018442765;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.287653446197511542) ) ) {
                    result[0] += -0.1034709397612516;
                  } else {
                    result[0] += 0.06462531816664413;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.005146580305778817;
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              result[0] += 0.03421561154432308;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.12471264051756432;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.013018911846622593;
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.0714328907929459;
                      } else {
                        result[0] += 0.002884412668310558;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.012107862492672719;
                    } else {
                      result[0] += -0.06252576621643509;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.003543875519454117;
                } else {
                  result[0] += 0.10796746299069815;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
              result[0] += 0.010480413996999992;
            } else {
              result[0] += -0.026165531865393367;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205624103546144354) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.278613805770874912) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0008463428233140235;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.48738741874694913) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                    result[0] += 0.013052948901025442;
                  } else {
                    result[0] += -0.00905665073615841;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += -0.04510734957918921;
                  } else {
                    result[0] += 0.008213782791182319;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += -0.02845487578375279;
              } else {
                result[0] += 0.009661128540414995;
              }
            }
          } else {
            result[0] += 0.0051063932955707275;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
          result[0] += -0.017327436514223287;
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.006362723653184227;
            } else {
              result[0] += -0.035399327978390495;
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.026945668136759932;
              } else {
                result[0] += 0.004751949321165745;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.01760628160784491;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.02187772927477522;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.011725283634000645;
                        } else {
                          result[0] += -0.03284883895168871;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.010386015275972224;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                      result[0] += -0.028686334334963395;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.04026887144413946;
                      } else {
                        result[0] += 0.002703012415004818;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.030123938173210435;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                          result[0] += 0.057720141842766595;
                        } else {
                          result[0] += -0.006030713990015843;
                        }
                      } else {
                        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                              result[0] += -0.019194890246626354;
                            } else {
                              result[0] += -0.00025877660294485616;
                            }
                          } else {
                            result[0] += -0.0423003422442256;
                          }
                        } else {
                          result[0] += -0.05272799609818819;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.00045928324813577305;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        result[0] += -0.010480694280093777;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
          result[0] += 0.00040969927855159295;
        } else {
          result[0] += -0.007450896261846834;
        }
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
            result[0] += -0.07341430389856683;
          } else {
            result[0] += 0.02035240161213615;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.848652839660646308) ) ) {
                result[0] += -0.022968039352012238;
              } else {
                result[0] += -0.11219630236503851;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.022574862598245253;
                  } else {
                    result[0] += -0.05396014054205866;
                  }
                } else {
                  result[0] += -0.060080148892745294;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.030515264684241268;
                } else {
                  result[0] += -0.04792646956777549;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.0017808566568161311;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                    result[0] += 0.005778805507939192;
                  } else {
                    result[0] += -0.018064974209692313;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.005801695422875541;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.048338733884247836;
                    } else {
                      result[0] += -0.013580844558276054;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                  result[0] += -0.010025878829448205;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.002473521433306055;
                  } else {
                    result[0] += 0.013236959848075389;
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.023019744203790007;
      }
    }
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += 0.003526705861313385;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
          result[0] += 0.004582312492919905;
        } else {
          result[0] += -0.05461865172155614;
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.047389297457135666;
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.441542863845826083) ) ) {
            result[0] += 0.019391609284180975;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.01636191019342287;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.067782521247864214) ) ) {
                    result[0] += -0.07469760013070129;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                        result[0] += -0.00690819814878228;
                      } else {
                        result[0] += 0.05329486119294266;
                      }
                    } else {
                      result[0] += -0.018435879847807157;
                    }
                  }
                }
              } else {
                result[0] += -0.00025983649623619645;
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
                  result[0] += -0.012905703252913578;
                } else {
                  result[0] += -0.06070041903148421;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.02139841801097222;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.038837455248149015;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                          result[0] += -0.08193375808777066;
                        } else {
                          result[0] += 0.0008284530592579178;
                        }
                      } else {
                        result[0] += 0.013108265683628812;
                      }
                    }
                  }
                } else {
                  result[0] += -0.023228611522295883;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
            result[0] += -0.0016383030543181743;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.05564652333033951;
                    } else {
                      result[0] += -0.048825446825191875;
                    }
                  } else {
                    result[0] += 0.031684030244199454;
                  }
                } else {
                  result[0] += -0.07119048040875701;
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.749434947967529741) ) ) {
                    result[0] += -0.09650149155413759;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.006600638285929815;
                      } else {
                        result[0] += 0.07438917692882147;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                        result[0] += -0.015917638415838885;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                          result[0] += -0.004024034032504643;
                        } else {
                          result[0] += 0.030161648240716262;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.436638832092286933) ) ) {
                        result[0] += 0.053926874454150724;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                          result[0] += -0.19217716951587532;
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.035378204354391075;
                          } else {
                            result[0] += -0.07059849439568679;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.10069523213982497;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                      result[0] += -0.004698221047952082;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.010698967846805256;
                      } else {
                        result[0] += 0.060342383061388685;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.026230716069451876;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.045098739036272206;
                } else {
                  result[0] += -0.004958159796941123;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.0006988195712450941;
  } else {
    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01340459395629734;
              } else {
                if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 0.03259565619636773;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.06134069360650296;
                  } else {
                    result[0] += 0.0003015526132857738;
                  }
                }
              }
            } else {
              result[0] += 0.008947969187420773;
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.007574898370381267;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.019539980689370365;
                  } else {
                    result[0] += -0.11479717194861434;
                  }
                }
              } else {
                result[0] += 0.017458711408267137;
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.02588922012160272;
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.016810894012452948) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += 0.07241402184052544;
                    } else {
                      result[0] += 0.016405232210487355;
                    }
                  } else {
                    result[0] += -0.005191752455238383;
                  }
                } else {
                  result[0] += -0.0224203682952597;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.843275547027588779) ) ) {
                result[0] += 0.044768787666920606;
              } else {
                result[0] += 0.008523313849563053;
              }
            } else {
              result[0] += -0.0691349620476248;
            }
          } else {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.057953596115113193) ) ) {
                  result[0] += 0.04176417390255556;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.941534638404846635) ) ) {
                    result[0] += 0.014709006244233814;
                  } else {
                    result[0] += -0.05393446383601387;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.0043532221689126915;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
                        result[0] += 0.10897507187215111;
                      } else {
                        result[0] += 0.04228141689577338;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.07465314865112482) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11882066726684748) ) ) {
                        result[0] += -0.019990564571235436;
                      } else {
                        result[0] += -0.1064907379225985;
                      }
                    } else {
                      result[0] += 0.05399323358333036;
                    }
                  }
                } else {
                  result[0] += -0.006237326236813172;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.047665546658622;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.105651378631592685) ) ) {
                      result[0] += -0.013489101446687694;
                    } else {
                      result[0] += 0.07362495103323524;
                    }
                  } else {
                    result[0] += -0.00492875567972304;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
                    result[0] += -0.059975561541571625;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                      result[0] += 0.1444431169748977;
                    } else {
                      result[0] += -0.054223857836888836;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.980170249938965732) ) ) {
          result[0] += -0.007011913689196398;
        } else {
          result[0] += 0.028139112052067122;
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.230628252029419833) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
            result[0] += 0.016217471975024524;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += 0.02049784566117061;
            } else {
              result[0] += -0.028151135844976756;
            }
          }
        } else {
          result[0] += -0.045359784362186295;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.182021141052246982) ) ) {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                  result[0] += -0.049597765071985635;
                } else {
                  result[0] += 0.08139226134177929;
                }
              } else {
                result[0] += 0.11292719857468295;
              }
            } else {
              result[0] += -0.05910725859535196;
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.0016509468444338685;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.007957301224263096;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                    result[0] += -0.011622165209095313;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.597323656082154208) ) ) {
                      result[0] += -0.0033878811474551867;
                    } else {
                      result[0] += 0.09963882015708536;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.07053746518582557;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.03149202847086929;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.947818994522095615) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.465643882751465732) ) ) {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                                result[0] += -0.05094397254673089;
                              } else {
                                result[0] += -0.1938963378190531;
                              }
                            } else {
                              result[0] += 0.017356058358005695;
                            }
                          } else {
                            result[0] += 0.08442440231550516;
                          }
                        } else {
                          result[0] += -0.12154191807933912;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                        result[0] += 0.012965413817360228;
                      } else {
                        result[0] += 0.06360010792828011;
                      }
                    }
                  } else {
                    result[0] += -0.02174775692894579;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.424685239791871005) ) ) {
              result[0] += 0.0042551370364622134;
            } else {
              result[0] += 0.0227514893702882;
            }
          } else {
            result[0] += -0.009660225150875286;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.0006631678923761516;
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
        result[0] += 0.004913756452249832;
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          result[0] += -0.040895263253007685;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += 0.035347202387139044;
          } else {
            result[0] += -0.007627230143017397;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.182021141052246982) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += 0.026282016319864848;
                } else {
                  result[0] += 0.00044258378635087865;
                }
              } else {
                result[0] += -0.026363926484880812;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.56941866874694913) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.18134641647339045) ) ) {
                  result[0] += -0.004738924333981733;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.0021873942128387606;
                  } else {
                    result[0] += 0.03755884198421746;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.09584809646405673;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.057823190108864365;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.02152021782862447;
                        } else {
                          result[0] += 0.010531244889089348;
                        }
                      } else {
                        result[0] += -0.04522502841737117;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.03449405183906626;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.114787101745606357) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                                    result[0] += -0.0031158333563025106;
                                  } else {
                                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                                      result[0] += -0.08255409674136575;
                                    } else {
                                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                                        result[0] += 0.010896919383423803;
                                      } else {
                                        result[0] += -0.08527281372230378;
                                      }
                                    }
                                  }
                                } else {
                                  result[0] += 0.0101120047813474;
                                }
                              } else {
                                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += -0.0006545466639606999;
                                } else {
                                  result[0] += 0.04275339899939922;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.056313823969008414;
                              } else {
                                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += 0.03461039316213689;
                                } else {
                                  result[0] += -0.036956686741898274;
                                }
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12043190002441584) ) ) {
                              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.008882948787650965;
                              } else {
                                result[0] += -0.0342262606707624;
                              }
                            } else {
                              result[0] += 0.04496565827405215;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.654679536819458896) ) ) {
                              result[0] += 0.001581478266123016;
                            } else {
                              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                                result[0] += -0.009745056334181854;
                              } else {
                                result[0] += -0.029206645773443292;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                              result[0] += 0.0009307742620307124;
                            } else {
                              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                                  result[0] += -0.07255115531397528;
                                } else {
                                  result[0] += 0.05717577601682096;
                                }
                              } else {
                                result[0] += -0.0043272820017932695;
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
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.424685239791871005) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03260980733582061;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.03554558943917934;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.039406418126620664;
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.028782470282284613;
                    } else {
                      result[0] += -0.006899786472018582;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.025818434088891062;
              } else {
                result[0] += 0.02079479433406645;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0055371431304480495;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.07071399845758045;
                    } else {
                      result[0] += 0.023344167085299797;
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.11906609627583993;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.10722307170779484;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                          result[0] += -0.06702015011802387;
                        } else {
                          result[0] += 0.15146672227261287;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.012850710274208208;
                }
              } else {
                result[0] += -0.0026482394257525605;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.051494711334812565;
            } else {
              result[0] += -0.009274742156699526;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += 0.000726596073771077;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += -0.034668454803051856;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.0032152351506876662;
                } else {
                  result[0] += -0.03267294523274659;
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.005758653484177234;
                } else {
                  result[0] += -0.051154414040908515;
                }
              } else {
                result[0] += -0.03214131781434125;
              }
            }
          }
        } else {
          result[0] += 0.011826684019370894;
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.0006539543906186767;
  } else {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.78508520126342951) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.021443025728749204;
            } else {
              result[0] += -0.026141228430295333;
            }
          } else {
            result[0] += -0.1345664678843284;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
            result[0] += 0.052262648742522955;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.06077854382449632;
              } else {
                result[0] += -0.0015094176506990254;
              }
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.005836081911200163;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.287653446197511542) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08565764785302833;
                  } else {
                    result[0] += -0.025366659108097836;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.623839378356934482) ) ) {
                    result[0] += -0.022525427569261222;
                  } else {
                    result[0] += 0.0613413103612199;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.042060907625054364;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.56941866874694913) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.01407949491550374;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
                      result[0] += -0.021102624803854365;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += 0.029735410923894857;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.04426131098280023;
                        } else {
                          result[0] += -0.046565194031490086;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.07156518080902097;
                      } else {
                        result[0] += -0.007827733565736381;
                      }
                    } else {
                      result[0] += 0.009700703519452639;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += -0.01872645218693552;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.06614608842923943;
                      } else {
                        result[0] += -0.0064677522607802275;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0019316804701451401;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
              result[0] += 0.012324687309876297;
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.991406440734865058) ) ) {
                  result[0] += 0.0031845212581926617;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.441542863845826083) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.985194206237793857) ) ) {
                        result[0] += 0.014288772233153944;
                      } else {
                        result[0] += -0.09821027217110141;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                        result[0] += 0.12190563281475533;
                      } else {
                        result[0] += 0.03714518710061091;
                      }
                    }
                  } else {
                    result[0] += -0.022851867092119834;
                  }
                }
              } else {
                result[0] += -0.0009304030363332853;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
                result[0] += -0.00304804907973006;
              } else {
                result[0] += -0.05778207731377755;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                      result[0] += 0.016850272797979286;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.02421458224686629;
                      } else {
                        result[0] += 0.06609002322206646;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.005546312215747232;
                      } else {
                        result[0] += -0.11897520859531825;
                      }
                    } else {
                      result[0] += 0.0173669868353493;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.012015666144805232;
                  } else {
                    result[0] += -0.011409491501182381;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.03420138359069913) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                        result[0] += -0.1443256295047909;
                      } else {
                        result[0] += 0.003057636581690166;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.255587577819825107) ) ) {
                        result[0] += 0.02427231833637486;
                      } else {
                        result[0] += 0.08655509928301212;
                      }
                    }
                  } else {
                    result[0] += -0.026388289250151627;
                  }
                } else {
                  result[0] += -0.03257448979600731;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11159896850586115) ) ) {
              result[0] += 0.0035316717092684644;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95053911209106623) ) ) {
                      result[0] += 0.03633824893704248;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.04530750505436639;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.035872205365038046;
                        } else {
                          result[0] += -0.11638195362048676;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.011402965157894188;
                  }
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.02225024913238302;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.022665386651172098;
                    } else {
                      result[0] += 0.05873538854783342;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.743881702423096591) ) ) {
                  result[0] += -0.02709425145690797;
                } else {
                  result[0] += 0.011585620772370452;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
        result[0] += -0.002708872209642036;
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
          result[0] += 0.23157846004708943;
        } else {
          result[0] += -0.03856978913624714;
        }
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.09844286538181354;
          } else {
            result[0] += -0.0216075927240769;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.363266706466675693) ) ) {
              result[0] += 0.005953969736957076;
            } else {
              result[0] += -0.015376140005401072;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.01071321143939235;
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.179772853851319248) ) ) {
                    result[0] += 0.0024866391856504374;
                  } else {
                    result[0] += -0.021634108274145598;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.206746339797974521) ) ) {
                        result[0] += -0.0010370291358027604;
                      } else {
                        result[0] += -0.04320406560661766;
                      }
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.893023490905762607) ) ) {
                              result[0] += -0.030275886247905476;
                            } else {
                              result[0] += -0.0635457139853991;
                            }
                          } else {
                            result[0] += -0.012664146714445115;
                          }
                        } else {
                          result[0] += -0.009054991690948276;
                        }
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.665046453475953037) ) ) {
                            result[0] += 0.09828369595977857;
                          } else {
                            result[0] += -0.01213070549546462;
                          }
                        } else {
                          result[0] += -0.01715275575216006;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0004532237365204317;
                  }
                }
              } else {
                result[0] += -0.030882461158518254;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
            result[0] += -0.1356644700585197;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += 0.06822581576155311;
            } else {
              result[0] += -0.03644918647420194;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += -0.0035567691709874577;
                    } else {
                      result[0] += -0.10779040902613768;
                    }
                  } else {
                    result[0] += 0.003412759655870366;
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.007996588881674708;
                      } else {
                        result[0] += -0.1574703536445071;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                        result[0] += -0.0034236295666857743;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.029964287324831657;
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                              result[0] += 0.03167463718739243;
                            } else {
                              result[0] += 0.10256850413988257;
                            }
                          }
                        } else {
                          result[0] += 0.0802441485878082;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                      result[0] += 0.031334866085020126;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.008296324208441851;
                      } else {
                        result[0] += -0.0712630656729087;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.047435612959291225;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.233438730239869052) ) ) {
                    result[0] += 0.003918947584495662;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
                      result[0] += -0.009777027978394488;
                    } else {
                      result[0] += 0.07894339011870202;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.152389049530031073) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                  result[0] += -0.0776654783194596;
                } else {
                  result[0] += -0.008044644424693215;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22084951400757014) ) ) {
                    result[0] += -0.059546871357435874;
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.038908246167752514;
                    } else {
                      result[0] += -0.03643849700459977;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    result[0] += -0.008909083364757554;
                  } else {
                    result[0] += 0.047384537337858905;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.006907109841568779;
            } else {
              result[0] += -5.7598260540292635e-05;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.0045408177643540765;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.05808441184784013;
              } else {
                result[0] += 0.012794520585416714;
              }
            } else {
              result[0] += -0.014443988837048411;
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
              result[0] += -0.028195019750380843;
            } else {
              result[0] += 0.005336295505300505;
            }
          }
        }
      } else {
        result[0] += 0.00021294996695159766;
      }
    }
  } else {
    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
          result[0] += 0.06887654962862115;
        } else {
          result[0] += -0.023352678313430422;
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
          result[0] += -0.04174995617024746;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
            result[0] += 0.07308356082758681;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += 0.017383462314875208;
            } else {
              result[0] += 0.15898764818993916;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
          result[0] += 0.013718397211653336;
        } else {
          result[0] += -0.06708632329501081;
        }
      } else {
        result[0] += -0.04746928898356527;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += 0.0004677112418815007;
  } else {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.051747083663941318) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.006246350474590152;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.01119236371955797;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0007574482358851191;
                  } else {
                    result[0] += -0.03226821117261826;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += 0.048345640092544134;
                    } else {
                      result[0] += -0.010159311960054679;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.028240149026667785;
                    } else {
                      result[0] += 0.008425393942728038;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += -0.004678972839962685;
                } else {
                  result[0] += 0.02082890579860013;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.017009537052037237;
                  } else {
                    result[0] += -0.059298513938815446;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.01259929177831522;
                      } else {
                        result[0] += 0.03187048559662152;
                      }
                    } else {
                      result[0] += -0.03184582190279738;
                    }
                  } else {
                    result[0] += -0.039855601797160134;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                result[0] += 0.019566172717982028;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0353464594471208;
                  } else {
                    result[0] += -0.017853539051406183;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.009621810617087036;
                    } else {
                      result[0] += -0.06436269603200552;
                    }
                  } else {
                    result[0] += -0.040650400277104225;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
                    result[0] += -0.006364849330281218;
                  } else {
                    result[0] += -0.050472055327558024;
                  }
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.0002205922323918564;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                      result[0] += 0.01362161648258763;
                    } else {
                      result[0] += 0.07976107128191703;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.04727481997189852;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                      result[0] += 0.01611132419557818;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.422742605209351474) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                          result[0] += -0.018088820014497055;
                        } else {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += -0.025592158008353606;
                          } else {
                            result[0] += 0.047041709213365285;
                          }
                        }
                      } else {
                        result[0] += -0.03355259962810678;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.026065755836908285;
                    } else {
                      result[0] += 0.13455162106969987;
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.006857864663180057;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.10521773020874639;
                            } else {
                              result[0] += 0.004221586826191103;
                            }
                          } else {
                            result[0] += 0.06713569121310771;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                              result[0] += 0.015964513422088093;
                            } else {
                              result[0] += 0.04295111981322558;
                            }
                          } else {
                            result[0] += 0.04910011975789741;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                          result[0] += 0.03788355589571026;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                            result[0] += -0.03016729693454111;
                          } else {
                            result[0] += 0.06041816131115756;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
                          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                                result[0] += 0.0753321079245312;
                              } else {
                                result[0] += -0.028021792482586114;
                              }
                            } else {
                              result[0] += 0.04740427035346772;
                            }
                          } else {
                            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.777633190155030185) ) ) {
                              result[0] += 0.02658194525801269;
                            } else {
                              result[0] += 0.06896254744458287;
                            }
                          }
                        } else {
                          result[0] += 0.09620445356439217;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.010977583609665217;
              } else {
                result[0] += -0.013164368989412222;
              }
            } else {
              result[0] += -0.027039467356920976;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += 0.012209353145188883;
          } else {
            result[0] += -0.03304671512051633;
          }
        } else {
          result[0] += -0.04675135239369588;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.450390577316285068) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.018025755724595357;
          } else {
            result[0] += -0.04067790123125728;
          }
        } else {
          result[0] += -0.0011918486649060785;
        }
      } else {
        if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.026570837109350903;
        } else {
          result[0] += -0.017821599505957762;
        }
      }
    }
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0006085518686763637;
    } else {
      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.004657953932639663;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.749261140823365146) ) ) {
                result[0] += 0.01683525593183754;
              } else {
                result[0] += -0.012687057754136888;
              }
            }
          } else {
            result[0] += -0.000170331485712997;
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.02046206599515256;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.057953596115113193) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.03556372730502522;
                    } else {
                      result[0] += -0.011716121195155539;
                    }
                  } else {
                    result[0] += 0.01654114089384387;
                  }
                } else {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                            result[0] += 0.05003165558466987;
                          } else {
                            result[0] += -0.011390409183776979;
                          }
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.993164777755738193) ) ) {
                            result[0] += 0.021980290883875847;
                          } else {
                            result[0] += 0.06841571602972485;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.03615108083144887;
                        } else {
                          result[0] += 0.004927790426071641;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.006782171023225637;
                        } else {
                          result[0] += 0.0558023073769307;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.617236852645874912) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.020841662109930933;
                          } else {
                            result[0] += -0.05618153033511336;
                          }
                        } else {
                          result[0] += 0.033448116786224266;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.01145692690443996;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                          result[0] += -0.052220777546420266;
                        } else {
                          result[0] += 0.034847000792201516;
                        }
                      } else {
                        result[0] += -0.06930472758510808;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                        result[0] += -0.009501871888034665;
                      } else {
                        result[0] += -0.08006194474641268;
                      }
                    } else {
                      result[0] += 0.07847771433372397;
                    }
                  } else {
                    result[0] += 0.0007877879766066574;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.715336322784424716) ) ) {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.023623300107062123;
                      } else {
                        result[0] += -0.07645726209017423;
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.092439889907837802) ) ) {
                          result[0] += -0.048347055685308796;
                        } else {
                          result[0] += -0.010684293326440752;
                        }
                      } else {
                        result[0] += -0.0010127463629684395;
                      }
                    }
                  } else {
                    result[0] += -0.05876555631787325;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.06623819637822788;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.028649011919428525;
                    } else {
                      result[0] += -0.004539453499577108;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.141444921493531162) ) ) {
                    result[0] += -0.00658603193902985;
                  } else {
                    result[0] += -0.03879044365793993;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.010032781905402076;
                  } else {
                    result[0] += -0.008843708259623632;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.0950601349737611;
                            } else {
                              result[0] += 0.019104409037467206;
                            }
                          } else {
                            result[0] += -0.01409562534930142;
                          }
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.020660157397190744;
                          } else {
                            result[0] += -0.13230899355011097;
                          }
                        }
                      } else {
                        result[0] += 0.038511865043142274;
                      }
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.11591598462380187;
                      } else {
                        result[0] += 0.05423854294220544;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                        result[0] += -0.02700059721438718;
                      } else {
                        result[0] += 0.02486442904410751;
                      }
                    } else {
                      result[0] += -0.06103154482007268;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.04421499917303108;
            } else {
              result[0] += -0.03739187798515555;
            }
          } else {
            result[0] += -0.037845017431107186;
          }
        } else {
          if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.055075315124710915;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                result[0] += 0.03616174208806119;
              } else {
                result[0] += -0.0016076787702255279;
              }
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
                result[0] += 0.0044378227093070065;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.851041555404663974) ) ) {
                    result[0] += 0.017250442537373154;
                  } else {
                    result[0] += -0.040362720850632354;
                  }
                } else {
                  result[0] += -0.04595816511716915;
                }
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.13241104395422562;
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0005940511061312883;
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += 0.003521949602610142;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.74845767021179288) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06133204204833425;
                  } else {
                    result[0] += 0.022721507119563306;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
                      result[0] += 0.008094821147438914;
                    } else {
                      result[0] += -0.03363528857059996;
                    }
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.001928133617745026;
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.046861171722413886) ) ) {
                          result[0] += -0.0815324358817633;
                        } else {
                          result[0] += 0.07332227645121303;
                        }
                      } else {
                        result[0] += 0.010027427903638886;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                  result[0] += 0.0012125791051099469;
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    result[0] += 0.014593539268487572;
                  } else {
                    result[0] += 0.06951969827687869;
                  }
                }
              } else {
                result[0] += -0.012923212579563024;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.01575865330666174;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.102759599685669833) ) ) {
                      result[0] += -0.03212612176107366;
                    } else {
                      result[0] += 0.03979213667522685;
                    }
                  } else {
                    result[0] += 0.004058820813908812;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += -0.1252170979983614;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += -0.0285512069523448;
                          } else {
                            result[0] += 0.08357382888623349;
                          }
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.126885652542115146) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                              result[0] += 0.05549878454608579;
                            } else {
                              result[0] += 0.012799971721057008;
                            }
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35311269760132014) ) ) {
                              result[0] += 0.02809574574881978;
                            } else {
                              result[0] += -0.11488362191598052;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.041921615600587714) ) ) {
                        result[0] += 0.0215231286178929;
                      } else {
                        result[0] += -0.11757627922568165;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                      result[0] += 0.04508436136590437;
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.0823238304695601;
                      } else {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            result[0] += -0.061809848306150786;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                              result[0] += -0.009950087715564685;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
                                  result[0] += 0.007650020524176036;
                                } else {
                                  result[0] += -0.12546084508671612;
                                }
                              } else {
                                result[0] += 0.0544841235913653;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                              result[0] += -0.005926928759703049;
                            } else {
                              result[0] += -0.09350638028291364;
                            }
                          } else {
                            result[0] += 0.009800570201857595;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    result[0] += -0.025811928730551437;
                  } else {
                    result[0] += 0.00932730938120807;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                result[0] += -0.004475705303229461;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.031560592699957604;
                  } else {
                    result[0] += -0.09963051536294269;
                  }
                } else {
                  result[0] += -0.006589382245091271;
                }
              }
            }
          }
        } else {
          result[0] += -0.0278047486160527;
        }
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.48738741874694913) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                result[0] += 0.045377737442411296;
              } else {
                result[0] += -0.08833294835402178;
              }
            } else {
              result[0] += 0.06706154079974688;
            }
          } else {
            result[0] += 0.09612762651443087;
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.403187274932863104) ) ) {
                result[0] += 0.0009710789412683982;
              } else {
                result[0] += -0.043336099880443256;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.230628252029419833) ) ) {
                result[0] += 0.014541102615346066;
              } else {
                result[0] += -0.04211358934522254;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
              result[0] += -0.0005473353932984734;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.06453473122790976;
                    } else {
                      result[0] += -0.004560104606786691;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.04513265732085781;
                    } else {
                      result[0] += 0.002414117920033148;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.506659984588624823) ) ) {
                      result[0] += 0.05220173301862954;
                    } else {
                      result[0] += 0.015068185893587417;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.012687900440892748;
                    } else {
                      result[0] += -0.04321550144924882;
                    }
                  }
                }
              } else {
                result[0] += -0.02266199462899142;
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.13241104395422562;
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.000570090838417189;
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.142269611358644354) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.1360614525519343;
                } else {
                  result[0] += 0.01944855321180078;
                }
              } else {
                result[0] += 0.07423794949931357;
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.494223117828370029) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)12.00000000000000178) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                                result[0] += 0.012859311441566738;
                              } else {
                                result[0] += 0.08773398735086796;
                              }
                            } else {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.465643882751465732) ) ) {
                                result[0] += 0.05483628989461903;
                              } else {
                                result[0] += -0.0022711021427452398;
                              }
                            }
                          } else {
                            result[0] += 0.006460121208998099;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
                            result[0] += 0.002257806673846268;
                          } else {
                            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                              result[0] += -0.07464720098069678;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                                result[0] += 0.039114911668388416;
                              } else {
                                result[0] += -0.02637358974035909;
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.027070070525481833;
                      }
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            result[0] += -0.001328214434092533;
                          } else {
                            result[0] += 0.04534544941937501;
                          }
                        } else {
                          result[0] += 0.08353489058361109;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.015721709741015444;
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.69067406654357999) ) ) {
                                    result[0] += 0.06203221774697996;
                                  } else {
                                    result[0] += -0.06697998406625526;
                                  }
                                } else {
                                  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                                    result[0] += 0.028224592801717886;
                                  } else {
                                    result[0] += 0.007191244567644874;
                                  }
                                }
                              } else {
                                result[0] += -0.006972119248700001;
                              }
                            } else {
                              result[0] += -0.004804389738336888;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                              result[0] += -0.03291414110272574;
                            } else {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.44831323623657404) ) ) {
                                result[0] += 0.009509482358181167;
                              } else {
                                result[0] += 0.08627327916436288;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.04540781528333344;
                            } else {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                                result[0] += -0.029887489182784662;
                              } else {
                                result[0] += 0.06212053562052782;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                          result[0] += -0.0021247131980192476;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += -0.07491125877035204;
                            } else {
                              result[0] += 0.00982609315021659;
                            }
                          } else {
                            result[0] += -0.09795850556881681;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                          result[0] += -0.05532812848160865;
                        } else {
                          result[0] += 0.06233320576785189;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.041362389750914524;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.233438730239869052) ) ) {
                            result[0] += 0.012116892682639505;
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                              result[0] += 0.09365445619729675;
                            } else {
                              result[0] += -0.008193142582836061;
                            }
                          }
                        } else {
                          result[0] += -0.05386460602014007;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.0049131674389247286;
                    } else {
                      result[0] += -0.01521716615772528;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                      result[0] += 0.03341036055204812;
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.78508520126342951) ) ) {
                                result[0] += 0.05094418561732004;
                              } else {
                                result[0] += -0.16351778449842172;
                              }
                            } else {
                              result[0] += -0.12046812891947954;
                            }
                          } else {
                            result[0] += 0.010750221347865026;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.13904914807794633;
                            } else {
                              result[0] += -0.01598572904703865;
                            }
                          } else {
                            result[0] += 0.000613629914778603;
                          }
                        }
                      } else {
                        result[0] += -0.0029964372135928836;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.0040827143017582255;
              }
            }
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.015243105713903993;
                } else {
                  result[0] += -0.12378535395080477;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.03211814359860485;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.10662474345269396;
                  } else {
                    result[0] += -0.0025209868638756184;
                  }
                }
              }
            } else {
              result[0] += -0.046213536116089075;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
            result[0] += 0.06718661612652758;
          } else {
            result[0] += -0.03568240423964613;
          }
        }
      } else {
        result[0] += 0.000275115549596944;
      }
    }
  } else {
    result[0] += -0.13241104395422562;
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.705447435379029208) ) ) {
          result[0] += -0.00017796937096343466;
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.05545081184674708;
                } else {
                  result[0] += -0.007983890857463686;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.12013978005987758;
                } else {
                  result[0] += -0.014539751417781122;
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.015049209232531177;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                    result[0] += 0.08469461720617173;
                  } else {
                    result[0] += -0.005731143464194566;
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.003811975150419341;
                  } else {
                    result[0] += -0.05508603341193613;
                  }
                } else {
                  result[0] += 0.10604072363462513;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.255004644393921787) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.042612366583284975;
                } else {
                  result[0] += -0.02907912264355669;
                }
              } else {
                result[0] += 0.010883556446527037;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08545036654125428;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += 0.03434806999027223;
                } else {
                  result[0] += -0.07857181896843879;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.06071388229533277;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                  result[0] += 0.02258556006114608;
                } else {
                  result[0] += -0.0021109179126121338;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.002711047923153523;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.00016562533730982005;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.027178552107511435;
                        } else {
                          result[0] += -0.10871251144490889;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.04216354505465124;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.329314231872559482) ) ) {
                            result[0] += -0.021906294199268886;
                          } else {
                            result[0] += 0.021838906013411805;
                          }
                        } else {
                          result[0] += -0.04726698261894337;
                        }
                      } else {
                        result[0] += 0.13265320914313805;
                      }
                    } else {
                      result[0] += 0.01179420399649718;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.01935453591708582;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                              result[0] += -0.0007085504607005081;
                            } else {
                              result[0] += 0.10313283360550901;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.025520505961925184;
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.02930698518314196;
                              } else {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                                  result[0] += 0.01544998172352388;
                                } else {
                                  result[0] += -0.03946953681055895;
                                }
                              }
                            } else {
                              result[0] += 0.031717478137866316;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.015192768482462864;
                      }
                    } else {
                      result[0] += -0.010370832429787688;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.400584220886231357) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.005929227473558881;
                    } else {
                      result[0] += 0.012895943136976196;
                    }
                  } else {
                    result[0] += -0.09532892880014808;
                  }
                } else {
                  result[0] += -0.00037006261303429623;
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.01148909681190379;
                } else {
                  result[0] += -0.02182791557806786;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.05101914629649787;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.05876592748561077;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.013402711436908903;
                      } else {
                        result[0] += -0.0842081184773528;
                      }
                    } else {
                      result[0] += -0.00015250413170845746;
                    }
                  }
                } else {
                  result[0] += 0.007461901261739035;
                }
              }
            } else {
              result[0] += -0.06563274603155318;
            }
          } else {
            result[0] += -0.027397173526501284;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.285748958587648261) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.978102684020996982) ) ) {
            result[0] += 0.011305396682309267;
          } else {
            result[0] += -0.04835482936955473;
          }
        } else {
          result[0] += 0.11771894655246994;
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.05491497916714599;
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.03278583204540596;
            } else {
              result[0] += -0.04725958582466593;
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.062253197954960704;
            } else {
              result[0] += -0.0009128547831108692;
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.13241104395422562;
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
        result[0] += 0.024680947768898;
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
          result[0] += -0.017118087578339245;
        } else {
          result[0] += -0.06745930289464443;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.293085813522339311) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.007937935195310403;
              } else {
                result[0] += -0.07136542924800836;
              }
            } else {
              result[0] += 0.006992493514024392;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
              result[0] += 0.013697938440427819;
            } else {
              result[0] += -0.005988474889446875;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.795494556427002841) ) ) {
            result[0] += -0.0008129206271005302;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.06633172203753433;
            } else {
              result[0] += 0.07348521861993607;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
              result[0] += 0.007569251200862808;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                result[0] += 0.0051475118622484155;
              } else {
                result[0] += 0.06577152946168004;
              }
            }
          } else {
            result[0] += -0.02389016251742027;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.00012757199218370663;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.07465314865112482) ) ) {
                  result[0] += 0.0011644502727927596;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.011957787352671827;
                    } else {
                      result[0] += 0.017393472527920225;
                    }
                  } else {
                    result[0] += -0.09932773716675168;
                  }
                }
              } else {
                result[0] += -0.032435692133699276;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.03952989703543385;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.11806181525496931;
                } else {
                  result[0] += 0.01487105651731533;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.07077343421926888;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
              result[0] += -0.006736123891123498;
            } else {
              result[0] += -0.03376897951507515;
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
            result[0] += -0.04378799813841483;
          } else {
            result[0] += -0.11337223960340842;
          }
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
            result[0] += -0.01001630934847607;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.029532062442776025;
              } else {
                result[0] += 0.01992724031506885;
              }
            } else {
              result[0] += -0.026767144284957658;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += 0.027136831377153222;
            } else {
              result[0] += 0.0982038234968035;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30779838562011896) ) ) {
              result[0] += 0.0004426208893539307;
            } else {
              result[0] += 0.01672177145769779;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.002663522629219697;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.01393969713121164;
            } else {
              result[0] += -0.10767592239424084;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.03965711152884393;
              } else {
                result[0] += -0.07220300104205633;
              }
            } else {
              result[0] += 0.03611420044012676;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.102609157562256748) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.777633190155030185) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.153024196624756748) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.006308158275164658;
                      } else {
                        result[0] += 0.019734986145557354;
                      }
                    } else {
                      result[0] += -0.013831474485912244;
                    }
                  } else {
                    result[0] += 0.0003997645750270065;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.714776992797852451) ) ) {
                      result[0] += -0.0607219915105525;
                    } else {
                      result[0] += -0.009224388415189909;
                    }
                  } else {
                    result[0] += -0.01018191599744353;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                  result[0] += -0.034597087337403645;
                } else {
                  if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                      result[0] += 0.007614448704147726;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.04094180742595852;
                      } else {
                        result[0] += -0.00677965452364726;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.03030169646427307;
                    } else {
                      result[0] += 0.026109338397171722;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.0025124704210096305;
            }
          } else {
            result[0] += 0.002691328118183245;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0113970440183601;
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.70802879333496271) ) ) {
                result[0] += 0.013169895918427542;
              } else {
                result[0] += -0.010158930825264013;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                result[0] += -0.01902869136709573;
              } else {
                result[0] += 0.022887186335292495;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.634540319442749912) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                result[0] += 0.11083976465747698;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0679346590439363;
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.977453231811524326) ) ) {
                    result[0] += 0.017327134138823277;
                  } else {
                    result[0] += -0.12687450690115276;
                  }
                }
              }
            } else {
              result[0] += -0.042792649636165254;
            }
          } else {
            result[0] += -0.012304198181849985;
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.009096809747266397;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
              result[0] += 0.11346909086365278;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.04714939307361262;
              } else {
                result[0] += -0.016720134414147474;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
            result[0] += 0.008481064272152452;
          } else {
            result[0] += -0.0012969490461426343;
          }
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.009962165356576142;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                  result[0] += -0.046002013698366914;
                } else {
                  result[0] += 0.059532566703324734;
                }
              } else {
                result[0] += -0.05036552991041654;
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.992907285690308505) ) ) {
                result[0] += -0.0016007895872477548;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.0037606095033054852;
                } else {
                  result[0] += -0.047353525534062725;
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.02596920781163352;
                    } else {
                      result[0] += -0.06346081436080517;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      result[0] += -0.04246805343006831;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                        result[0] += 0.015702084923569976;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                            result[0] += -0.08609651457476228;
                          } else {
                            result[0] += -0.027541520693339257;
                          }
                        } else {
                          result[0] += -0.005165455537944476;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.004803528485546642;
                }
              } else {
                result[0] += 0.020987746762139808;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
            result[0] += 0.00039166396297851057;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
              result[0] += 0.016166070397281417;
            } else {
              result[0] += 0.060218133131378054;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.400584220886231357) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
              result[0] += 0.002077347308300658;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.002651223948246857;
              } else {
                result[0] += -0.0189045636386492;
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.006201348011321384;
            } else {
              result[0] += -0.027053723136195904;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.0004012369787080668;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.050975706147549975;
                } else {
                  result[0] += -0.008011905885057347;
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.941534638404846635) ) ) {
                    result[0] += -0.013740046989695151;
                  } else {
                    result[0] += 0.026456772099848747;
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += 0.0346479742990478;
                  } else {
                    result[0] += 0.11812688796075757;
                  }
                }
              }
            } else {
              result[0] += -0.048222308320938656;
            }
          }
        } else {
          result[0] += 0.002883247276003498;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
        result[0] += 0.03812905992496558;
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.04885300183495152;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.06009754247422392;
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.013028431195653818;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
                        result[0] += 0.02098717779132682;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.014434083555793826;
                        } else {
                          result[0] += 0.0965705036927767;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0325376907469782;
                    } else {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.007079250565292979;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.057674790364207196;
                        } else {
                          result[0] += 0.016397393980061965;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.07068018554813481;
              }
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
              result[0] += -0.010329721195038876;
            } else {
              result[0] += -0.05336833690778138;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
            result[0] += -0.1040268986020702;
          } else {
            result[0] += 0.060040618022997205;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.05413521472363071;
      } else {
        result[0] += 0.01784910256955938;
      }
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
      result[0] += 0.0062887962733315156;
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.993164777755738193) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.004511223798470155;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
            result[0] += -0.03422155234829782;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.025530716564845225;
            } else {
              result[0] += -0.004743636378312966;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.01680919995139397;
          } else {
            result[0] += -0.008065653055625918;
          }
        } else {
          result[0] += -0.03140982138203615;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      result[0] += -0.01534551954501619;
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.591613531112671787) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.04178524291865281;
              } else {
                result[0] += 0.043563907115626736;
              }
            } else {
              result[0] += -0.016304121787524665;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
              result[0] += -0.022884534448883394;
            } else {
              result[0] += 0.08165221559126455;
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
                        result[0] += 0.05127365182015875;
                      } else {
                        result[0] += -0.047936774147215394;
                      }
                    } else {
                      result[0] += 0.037646412507816654;
                    }
                  } else {
                    result[0] += -0.04148463773452609;
                  }
                } else {
                  result[0] += -0.1303934063644819;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.0014812750304633894;
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.04636925763125016;
                      } else {
                        result[0] += 0.023325668652530596;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.947025299072267401) ) ) {
                      result[0] += -0.05961723541136338;
                    } else {
                      result[0] += 0.010941642070760788;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.025093361127622206;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.020675656976999825;
                      } else {
                        result[0] += -0.11753343471771956;
                      }
                    }
                  } else {
                    result[0] += 0.03861586615627735;
                  }
                }
              }
            } else {
              result[0] += -0.04033756507962289;
            }
          } else {
            result[0] += 0.026388957071098382;
          }
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.0026969220983376965;
              } else {
                result[0] += -0.08692695606071103;
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.044831322982527386;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += -0.0518566528933597;
                } else {
                  result[0] += -0.013797595868296125;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                result[0] += 0.0011191851978418622;
              } else {
                result[0] += -0.023825040274090284;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                result[0] += 0.030275734793290196;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.016535815247919066;
                } else {
                  result[0] += -0.03270694183332747;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                    result[0] += -0.030001853962729714;
                  } else {
                    result[0] += 0.00848740530522162;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.044508718519630425;
                    } else {
                      result[0] += -0.006778526526698726;
                    }
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.010171314445200773;
                    } else {
                      result[0] += 0.04167759465320739;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.2807660102844256) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.10238678196094658;
                  } else {
                    result[0] += 0.036191295719750265;
                  }
                } else {
                  result[0] += -0.005232377509686373;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
                    result[0] += 0.10112363723066767;
                  } else {
                    result[0] += -0.010938214878121199;
                  }
                } else {
                  result[0] += 0.0008187347252821765;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    result[0] += 0.005720320921676237;
                  } else {
                    result[0] += -0.01585261644689151;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.69067406654357999) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.05741461436012495;
                    } else {
                      result[0] += -0.018914594618007378;
                    }
                  } else {
                    result[0] += 0.02649677219733453;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.011563642461780247;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                result[0] += -0.00469003449356496;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += 0.020714819173218782;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.009719132022454951;
                    } else {
                      result[0] += -0.03806851860146945;
                    }
                  } else {
                    result[0] += 0.0004918304368052025;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.09262754199818105;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += 0.001050041151568626;
            } else {
              result[0] += 0.03330628909134553;
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.043802261352539951) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.94957673549652144) ) ) {
              result[0] += 0.09423919732748189;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.5240359306335467) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                      result[0] += -0.11228685379589352;
                    } else {
                      result[0] += -0.0012581482567850185;
                    }
                  } else {
                    result[0] += -0.15290941702612107;
                  }
                } else {
                  result[0] += -0.19933980787756128;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                  result[0] += 0.08106738559392175;
                } else {
                  result[0] += -0.01455961139480784;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
              result[0] += -0.1791469190545464;
            } else {
              result[0] += 0.002880146813116321;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95053911209106623) ) ) {
                result[0] += 0.0039085069993623515;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.18965101242065607) ) ) {
                  result[0] += -0.16757327244097722;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                    result[0] += 0.05488016411040308;
                  } else {
                    result[0] += -0.11776470035680786;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.142630577087403232) ) ) {
                result[0] += 0.10734521939099105;
              } else {
                result[0] += 0.0014771231473368071;
              }
            }
          } else {
            result[0] += 0.03170272160597378;
          }
        } else {
          result[0] += -0.06266829607925149;
        }
      }
    } else {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += 0.0037555616861376;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.784468173980714667) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
              result[0] += 0.030579784347607487;
            } else {
              result[0] += -0.035942130096341056;
            }
          } else {
            result[0] += 0.08344083584969136;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.50166225433349787) ) ) {
            result[0] += -0.09955627987141091;
          } else {
            result[0] += -0.03442237359927943;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += -0.0716126846744296;
      } else {
        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.08537480840427554;
            } else {
              result[0] += -0.0007885457450257051;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += -0.05361785410371696;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                result[0] += 0.09945685728474343;
              } else {
                result[0] += 0.0383526981952447;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += -0.06959003440114586;
            } else {
              result[0] += 0.04158832159809553;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.624251961708069292) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += -0.09462802912244202;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.027796840988212057;
                    } else {
                      result[0] += 0.09503579391905835;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                      result[0] += 0.032323425207509845;
                    } else {
                      result[0] += -0.06780964665988783;
                    }
                  }
                }
              } else {
                result[0] += -0.11206530321488793;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0419118323519125;
              } else {
                result[0] += 0.03634589393812481;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
          result[0] += 0.004761465580215857;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.00017215847067763237;
            } else {
              result[0] += -0.0253439185400023;
            }
          } else {
            result[0] += -0.009988371801984702;
          }
        }
      } else {
        if ( UNLIKELY(  (data[37].missing != -1) && (data[37].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.009425969047884472;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.033785503370563845;
                } else {
                  result[0] += 0.04075599943180869;
                }
              } else {
                result[0] += -0.024314888159773937;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.876230478286744052) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                      result[0] += 0.01118446179115744;
                    } else {
                      result[0] += -0.10257694524988947;
                    }
                  } else {
                    result[0] += -0.09215237242432789;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.006221888933410007;
                  } else {
                    result[0] += -0.07637899382846577;
                  }
                }
              } else {
                result[0] += 0.026753615829619076;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.0016564982524932235;
              } else {
                result[0] += 0.1563780751390423;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += -0.0030315444018113448;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.09728620915921406;
                  } else {
                    result[0] += -0.029021002267777193;
                  }
                }
              } else {
                result[0] += -0.004873198570565391;
              }
            }
          } else {
            result[0] += 7.07925171120486e-05;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.838940858840943271) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.08034600397107891;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.0009926775927294666;
              } else {
                result[0] += 0.032363290360286574;
              }
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.863673448562622958) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.467917680740357333) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.909102678298951083) ) ) {
                    result[0] += -0.09252277660874048;
                  } else {
                    result[0] += 0.010229313344428378;
                  }
                } else {
                  result[0] += -0.18670257069117288;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.993164777755738193) ) ) {
                  result[0] += 0.056435904032829935;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                      result[0] += -0.01671586935852096;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.021600330221138078;
                      } else {
                        result[0] += 0.07422927395248217;
                      }
                    }
                  } else {
                    result[0] += -0.06049632061025143;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                result[0] += -0.01853490616263539;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  result[0] += -0.22018677639094195;
                } else {
                  result[0] += -0.07889645920325832;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.923617362976075107) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.040285587310792792) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.427441358566285068) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                        result[0] += -0.03215559436609732;
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.054943854747216386;
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81940793991089045) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                              result[0] += -0.023693201387555166;
                            } else {
                              result[0] += 0.10294667690823862;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.43267917633056818) ) ) {
                              result[0] += -0.1816151640383451;
                            } else {
                              result[0] += -0.007587486838969343;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.09195845980379459;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.029571074071558998;
                          } else {
                            result[0] += -0.10622968634150616;
                          }
                        } else {
                          result[0] += -0.0016623289380531426;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                        result[0] += -0.044456654223764616;
                      } else {
                        result[0] += 0.0584932208966191;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                        result[0] += 0.11085188344673945;
                      } else {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                          result[0] += 0.017047922399489168;
                        } else {
                          result[0] += -0.03447134251707943;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.09093437174137387;
                }
              } else {
                result[0] += -0.048022707295836006;
              }
            } else {
              result[0] += 0.05268514933170611;
            }
          } else {
            result[0] += -0.05220063621773529;
          }
        }
      } else {
        result[0] += 0.1503783585994735;
      }
    } else {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.164715528488160068) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              result[0] += -0.05918617728392206;
            } else {
              result[0] += 0.012489472248302866;
            }
          } else {
            result[0] += 0.004694652167348693;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.06993949815504386;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
              result[0] += 0.12099054335063841;
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.009763779251987745;
              } else {
                result[0] += 0.06786329253057577;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.880305767059327948) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.743881702423096591) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
              result[0] += 0.012097252814389996;
            } else {
              result[0] += 0.07554661660333867;
            }
          } else {
            result[0] += -0.045103966676773945;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
            result[0] += 0.007962609553752917;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
              result[0] += -0.11047404976672966;
            } else {
              result[0] += -0.0484426919706862;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
          result[0] += 0.023159921434858705;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
            result[0] += 0.007494432982069131;
          } else {
            result[0] += -0.041743553288378525;
          }
        }
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += 0.0022743881506495952;
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.337269306182862216) ) ) {
            result[0] += -0.0015331506808963584;
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.002715381120262001;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.531673669815064365) ) ) {
                  result[0] += -0.008208381052657303;
                } else {
                  result[0] += -0.04408054802992932;
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.014964618146347764;
                } else {
                  result[0] += 0.03413586260216001;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += 0.004643901415167354;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.0013002103409261135;
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.36986422538757413) ) ) {
                result[0] += 0.0016377937355720664;
              } else {
                result[0] += -0.034049227097765984;
              }
            } else {
              result[0] += -0.005164895446535288;
            }
          }
        }
      } else {
        result[0] += 0.001447271151004734;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.0006948091171568838;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.617236852645874912) ) ) {
              result[0] += -0.06948510430988895;
            } else {
              result[0] += 0.06101400213983833;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.002570321059606718;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.011866981973117426;
                    } else {
                      result[0] += -0.058975714057530396;
                    }
                  } else {
                    result[0] += -0.06303330101408074;
                  }
                } else {
                  result[0] += 0.021199418317582718;
                }
              }
            } else {
              result[0] += -0.036285558555755784;
            }
          }
        }
      } else {
        result[0] += 0.0022862530531204115;
      }
    } else {
      result[0] += -0.006990972571514889;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      result[0] += 0.0004975914960853779;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.060294389724732333) ) ) {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06758670427035117;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                    result[0] += -0.07376941154923049;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.07940751123652545;
                    } else {
                      result[0] += 0.03887790418205173;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)257681260544.0000305) ) ) {
                      result[0] += 0.03657417963665162;
                    } else {
                      result[0] += -0.003998672067052035;
                    }
                  } else {
                    result[0] += 0.04340767169098885;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
                    result[0] += 0.00708163456615893;
                  } else {
                    result[0] += -0.0653323627662536;
                  }
                } else {
                  result[0] += -0.01857783117926599;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                    result[0] += -0.022352911725214007;
                  } else {
                    result[0] += 0.021298489523365747;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += -0.010509857280866805;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.06361260463633771;
                    } else {
                      result[0] += -0.0016779577972314587;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.022038129998109737;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.511434078216553178) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0002436004733149839;
                } else {
                  result[0] += 0.16824560300835206;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.70802879333496271) ) ) {
                  result[0] += 0.018263005009402655;
                } else {
                  result[0] += -0.005496975361379718;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                    result[0] += 0.004703659963033941;
                  } else {
                    result[0] += -0.015382731629513473;
                  }
                } else {
                  result[0] += -0.03654810712053442;
                }
              } else {
                result[0] += -0.029414471493332036;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.665476083755494052) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.028105782051184188;
                    } else {
                      result[0] += 0.014207435042516207;
                    }
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.008033966592319556;
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.028824978790638695;
                      } else {
                        result[0] += -0.04013527053250273;
                      }
                    }
                  }
                } else {
                  result[0] += 0.004129562138118248;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.07792926489919501;
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.0016416542263018139;
                      } else {
                        result[0] += -0.05125238320098784;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.07193629973734642;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
                            result[0] += 0.04545079645773498;
                          } else {
                            result[0] += -0.17392601958837856;
                          }
                        } else {
                          result[0] += -0.04367568712242249;
                        }
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.0428672159241902;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.005852194856393636;
                            } else {
                              result[0] += 0.050979190123431384;
                            }
                          } else {
                            result[0] += 0.0030833220208474677;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += -0.02405839989344591;
                  } else {
                    result[0] += 0.007590409768542566;
                  }
                }
              }
            }
          } else {
            result[0] += 0.013134016010974418;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
              result[0] += -0.02008629549553701;
            } else {
              result[0] += -0.00631557587581932;
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.02053902149172582;
            } else {
              result[0] += -0.06340330506510918;
            }
          }
        } else {
          if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06123815216928272;
            } else {
              result[0] += 0.030006041295436582;
            }
          } else {
            result[0] += -0.013694895806114118;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += 0.033787198557017666;
                } else {
                  result[0] += -0.06327514404113133;
                }
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.016650664804456386;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
                    result[0] += 0.0935123133345079;
                  } else {
                    result[0] += 0.02521590709202592;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.0063547835795028755;
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                  result[0] += -0.13340227332510185;
                } else {
                  result[0] += -0.016603089718845335;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.971558809280396396) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.060294389724732333) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.007950825036198676;
                    } else {
                      result[0] += -0.030747307598293774;
                    }
                  } else {
                    result[0] += 0.0008673890950085905;
                  }
                } else {
                  result[0] += 0.017739087585374146;
                }
              } else {
                result[0] += -0.17314846679923873;
              }
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.003356054627766706;
              } else {
                result[0] += -0.0498721545452496;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.0024420433388380293;
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.166635274887085849) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.002328236133655243;
                } else {
                  result[0] += 0.010446124177908099;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.003127408741120991;
                } else {
                  result[0] += 0.028625822643650406;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0787106284399216;
                } else {
                  result[0] += -0.005777900806248634;
                }
              } else {
                result[0] += -0.01780540287068181;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
          result[0] += 0.0014489675039825573;
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.807895898818970615) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.992907285690308505) ) ) {
                result[0] += -0.010337226838205371;
              } else {
                result[0] += -0.03896440761702351;
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.07059600134772553;
                } else {
                  result[0] += 0.012553383173052846;
                }
              } else {
                result[0] += -0.10210600457702688;
              }
            }
          } else {
            result[0] += 0.00037897425364803765;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
        result[0] += 0.0004789672238411527;
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.714269638061524326) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
              result[0] += 0.0011663445625258017;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                result[0] += 0.010412883612958715;
              } else {
                result[0] += -0.10235609973955231;
              }
            }
          } else {
            result[0] += -0.06564406749140633;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.05435905330570069;
            } else {
              result[0] += -0.00798789637206031;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.08934589463502816;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                    result[0] += 0.021075015158538647;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.026159883632937583;
                    } else {
                      result[0] += -0.034066534161846206;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.664693593978882724) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.05746846015874037;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 5.786067182944143e-05;
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.056733823035136834;
                        } else {
                          result[0] += 0.014330310699686556;
                        }
                      }
                    } else {
                      result[0] += 0.029714257019978846;
                    }
                  }
                } else {
                  result[0] += 0.06378449176470079;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.014887887399337907;
              } else {
                result[0] += -0.15149311695362896;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)1.151292562484741433) ) ) {
        result[0] += 0.014583787194779636;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.00610273282073839;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.302512168884278232) ) ) {
                result[0] += -0.025720163381144263;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.007632412422366153;
                      } else {
                        result[0] += 0.05422834861040224;
                      }
                    } else {
                      result[0] += -0.01217635164180807;
                    }
                  } else {
                    result[0] += -0.024835835248102567;
                  }
                } else {
                  result[0] += -0.03274595575981867;
                }
              }
            } else {
              result[0] += 0.0150451572686462;
            }
          }
        } else {
          result[0] += -0.037783748388474064;
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.553712725639343706) ) ) {
        result[0] += -0.00023928479996288148;
      } else {
        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.08136398655365991;
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.004851723518653305;
          } else {
            result[0] += -0.06769507206901916;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.24121904373169123) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 6.099867917488678e-05;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22084951400757014) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.15836188516825347;
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.025779174001959823;
                  } else {
                    result[0] += -0.0689825099190846;
                  }
                }
              } else {
                result[0] += 0.005446254958079655;
              }
            } else {
              result[0] += 0.03651743323952658;
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.001618542482225029;
            } else {
              result[0] += -0.024239077083469233;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
          result[0] += -0.0033109744312986716;
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.247576236724854404) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.0160573507810471;
              } else {
                result[0] += -0.12266243700110398;
              }
            } else {
              result[0] += -0.004274497135142573;
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.008739199613151581;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.497097015380861151) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.04741187284047081;
                      } else {
                        result[0] += 0.007480810631692396;
                      }
                    } else {
                      result[0] += -0.000957162790735927;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                        result[0] += -0.005128448550930086;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.018932007757318847;
                        } else {
                          result[0] += -0.07150471656480555;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.474771499633789951) ) ) {
                        result[0] += -0.007443441078148457;
                      } else {
                        result[0] += 0.0292957890822954;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.450390577316285068) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.020965608053102602;
                      } else {
                        if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.008029665208956342;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                            result[0] += -0.030452409062438086;
                          } else {
                            result[0] += 0.04590436892527704;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.02906081884683137;
                    }
                  } else {
                    result[0] += 0.007633825830175205;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.005480220418347447;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.03572678734699683;
                      } else {
                        result[0] += -0.08224374236994866;
                      }
                    } else {
                      result[0] += 0.0863382278919086;
                    }
                  } else {
                    result[0] += 0.012897936572217623;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.0546966034987513;
                  } else {
                    result[0] += 0.02423656697945269;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.006739825617368498;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.02807260798249536;
          } else {
            result[0] += -0.03324218210083702;
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.0053502031934328275;
          } else {
            result[0] += -0.027314898755153347;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          result[0] += 0.004367702296996368;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
            result[0] += 0.0009515716892107613;
          } else {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.267844915390015537) ) ) {
                      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05814635261098097;
                      } else {
                        result[0] += 0.004548568176189858;
                      }
                    } else {
                      result[0] += -0.013375466673343063;
                    }
                  } else {
                    result[0] += 0.0071219058648442695;
                  }
                } else {
                  result[0] += 0.08159749747946499;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.01053306569840587;
                  } else {
                    if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.06434726609565054;
                    } else {
                      result[0] += 0.019912642159889995;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.031109288122697778;
                    } else {
                      result[0] += -0.07336694295956582;
                    }
                  } else {
                    result[0] += 0.038325983989968476;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.004715784911789747;
                    } else {
                      result[0] += -0.019965533021296347;
                    }
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.002458028421610735;
                    } else {
                      result[0] += -0.042957495226760026;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.262283086776734287) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.004861630537853415;
                    } else {
                      result[0] += 0.06823596543436442;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.329314231872559482) ) ) {
                      result[0] += 0.009104829690602416;
                    } else {
                      result[0] += -0.0025757999853686965;
                    }
                  }
                }
              } else {
                result[0] += -0.025608896099821584;
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.006221716049236498;
      } else {
        result[0] += -0.026825748216590763;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59476566314697443) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.816582441329956943) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.594915628433228427) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0006734974996863261;
            } else {
              result[0] += -0.009392851004957656;
            }
          } else {
            result[0] += 0.0019095238682434069;
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.012458528416925217;
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.0008702345695701362;
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.014700430905708446;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                        result[0] += -0.03666071028039125;
                      } else {
                        result[0] += -0.11371729548395791;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.534971714019776279) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.628555774688722479) ) ) {
                        result[0] += -0.02904697536577131;
                      } else {
                        result[0] += 0.008986708041594677;
                      }
                    } else {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += 0.05335514685865655;
                      } else {
                        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.060711768639787715;
                        } else {
                          result[0] += 0.014172265006591231;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04901341107907439;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.02173928258888029;
              } else {
                result[0] += 0.015115256526129698;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.11324629705231944;
            } else {
              result[0] += -0.05497880125208704;
            }
          }
        }
      } else {
        result[0] += 0.008781194576365467;
      }
    } else {
      result[0] += -0.006068734879023932;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
      result[0] += 0.00044365572661117264;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.532332420349121982) ) ) {
              result[0] += 0.002078719479678527;
            } else {
              result[0] += -0.02668050464691554;
            }
          } else {
            result[0] += 0.13063325422240699;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.909254074096680576) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.02383757437473568;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
                      result[0] += 0.017718625845791612;
                    } else {
                      result[0] += -0.018767263811578224;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.014620839240732268;
                  } else {
                    result[0] += -0.03485069771993328;
                  }
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.010628961701014692;
                    } else {
                      result[0] += -0.020013156570462928;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.014662333072515242;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.013053437361937696;
                        } else {
                          if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.06903008191247861;
                          } else {
                            result[0] += -0.1026712346833416;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.65754175186157404) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.219419956207276279) ) ) {
                            result[0] += -0.012014980521863486;
                          } else {
                            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += 0.007943025392777933;
                            } else {
                              result[0] += 0.04843594169145682;
                            }
                          }
                        } else {
                          result[0] += 0.046850015968552365;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    result[0] += 0.0036083111911199544;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.092439889907837802) ) ) {
                      result[0] += -0.003976195805662416;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.028043022321090252;
                      } else {
                        result[0] += -0.06560667998301044;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.015943119124146716;
              } else {
                if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0885774558697711;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.827801465988160068) ) ) {
                    result[0] += -0.04260041779419948;
                  } else {
                    result[0] += 0.015501053319270451;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.004527394784378683;
              } else {
                result[0] += -0.0602606457178008;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.02439028060241338;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.009910189153884166;
                } else {
                  result[0] += 0.0550955760346839;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.0173301432922693;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.507949829101563388) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.029068946838379794) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                    result[0] += -0.001537812016262623;
                  } else {
                    result[0] += -0.04466238484364676;
                  }
                } else {
                  result[0] += 0.010553450644009116;
                }
              } else {
                result[0] += 0.050379915090636655;
              }
            } else {
              result[0] += 0.12023924860449241;
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
              result[0] += -0.013596645418606605;
            } else {
              result[0] += -0.2088946348680133;
            }
          } else {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.035591621146131534;
              } else {
                result[0] += 0.029074555416804895;
              }
            } else {
              result[0] += -0.05039324555036743;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.426736354827881748) ) ) {
        result[0] += 0.001159634546875928;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.01884510679279785;
            } else {
              result[0] += -0.030104804520856877;
            }
          } else {
            result[0] += -0.06933670368004671;
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.012247800794005302;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.020640477267192245;
              } else {
                result[0] += 0.008392002372740966;
              }
            }
          } else {
            result[0] += -0.06260952839685713;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.880305767059327948) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
          result[0] += -0.004816112813534259;
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.01696377610199407;
          } else {
            result[0] += -0.04815475014686683;
          }
        }
      } else {
        result[0] += -0.0009223779224265728;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.547126770019532138) ) ) {
              result[0] += 0.0006805321438317237;
            } else {
              result[0] += 0.03108252191583844;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.007350070786677484;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.94957673549652144) ) ) {
                result[0] += 0.03328698509823491;
              } else {
                result[0] += 0.00873111968709674;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              result[0] += -0.028345448501930482;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.03331717278988601;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.08200179033309259;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.334978580474854404) ) ) {
                            result[0] += -0.038508500003977715;
                          } else {
                            result[0] += 0.007368359611423213;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.00404496456719268;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.737603187561036044) ) ) {
                      result[0] += 0.015523889250305276;
                    } else {
                      result[0] += 0.09780455721891533;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.011457756480019001;
                    } else {
                      result[0] += 0.018772046417825955;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.021024379265183055;
                      } else {
                        result[0] += 0.0008772991683069103;
                      }
                    } else {
                      result[0] += -0.04000632157570865;
                    }
                  }
                }
              } else {
                result[0] += 0.013769753635607408;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.019657877380984484;
                } else {
                  result[0] += -0.05346566332212449;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.011397852349509636;
                } else {
                  result[0] += -0.02408144868828677;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.07804213075893891;
                } else {
                  result[0] += 0.0005911324931077631;
                }
              } else {
                result[0] += 0.01225214230804799;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
                result[0] += -0.0059816374230465385;
              } else {
                result[0] += -0.107410926004128;
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.03489323016751746;
              } else {
                result[0] += -0.013253721906964689;
              }
            }
          } else {
            result[0] += -0.020055138974010265;
          }
        } else {
          result[0] += -0.02437719526723834;
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.51492977142334162) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.009501954014171853;
              } else {
                result[0] += -0.0265465917720792;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += -0.000995222218292525;
              } else {
                result[0] += 0.036512651992951374;
              }
            }
          } else {
            result[0] += -0.02707621863070317;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
            result[0] += -0.02008233644554475;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.41263532638549982) ) ) {
                result[0] += 0.0017802043108974932;
              } else {
                result[0] += -0.01806614164585343;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      result[0] += -0.015030338168198848;
                    } else {
                      result[0] += -0.059049674206642325;
                    }
                  } else {
                    result[0] += -0.04859303224651679;
                  }
                } else {
                  if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0024514255741120163;
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.005827221902274405;
                    } else {
                      result[0] += -0.039492228723323475;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.0023820017921593597;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.006606974785634576;
                  } else {
                    result[0] += 0.104781705310039;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
          result[0] += -0.033338826455837856;
        } else {
          result[0] += -0.20789692937365387;
        }
      }
    }
  }
  if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0032892138794712667;
            } else {
              result[0] += 0.014080239841020255;
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.992712974548340732) ) ) {
                  result[0] += 0.0400036900014096;
                } else {
                  result[0] += -0.0021843484240906814;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.08248082027909703;
                } else {
                  result[0] += -0.007223884611715673;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.025250205539869203;
              } else {
                result[0] += -0.00041713041867319094;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.01635500226784804;
          } else {
            result[0] += -0.00038841409502491254;
          }
        }
      } else {
        result[0] += -0.010872086535575655;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.036423914046481064;
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.019332001646340483;
            } else {
              result[0] += 0.033299468898054775;
            }
          }
        } else {
          result[0] += 0.028246069172756502;
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.509355545043946201) ) ) {
                    result[0] += 0.019222471492524706;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.129040718078614169) ) ) {
                      result[0] += 0.01524189113521862;
                    } else {
                      if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.07853031455378157;
                      } else {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.497398376464845526) ) ) {
                            result[0] += 0.009012693533617637;
                          } else {
                            result[0] += -0.07088524428207935;
                          }
                        } else {
                          result[0] += 0.03796978164670817;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                    result[0] += -0.2093610008254317;
                  } else {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.0033878470820602803;
                    } else {
                      result[0] += 0.040266014602147335;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.381086945533752885) ) ) {
                    result[0] += 0.12165870225202052;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.003938685648496333;
                    } else {
                      result[0] += -0.029414342600419325;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.031997102588783674;
                  } else {
                    result[0] += 0.013648029730710762;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.614335536956787998) ) ) {
                  result[0] += -0.02741548280518647;
                } else {
                  result[0] += 0.02546839784982475;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.03072140299277249;
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06060445707189745;
                  } else {
                    result[0] += -0.0297576381596744;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.230628252029419833) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                    result[0] += -0.05607017422573797;
                  } else {
                    result[0] += -0.005732762267005889;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.06628829273073249;
                  } else {
                    result[0] += -0.016720340197974155;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
                    result[0] += -0.0624934901156817;
                  } else {
                    result[0] += 0.055413848287148106;
                  }
                } else {
                  result[0] += -0.09848664655363264;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.745876312255860263) ) ) {
                result[0] += 0.0021547821258702123;
              } else {
                result[0] += 0.07211170078033356;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.007688944072282156;
                  } else {
                    result[0] += -0.04003765906715285;
                  }
                } else {
                  result[0] += 0.006190476167053133;
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.009498980140640642;
                    } else {
                      result[0] += -0.0664199356596258;
                    }
                  } else {
                    result[0] += -0.03259646581142477;
                  }
                } else {
                  result[0] += -0.07754874272039004;
                }
              }
            } else {
              result[0] += -0.07178158779162225;
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.020027187330276378;
              } else {
                result[0] += 0.04640908639031772;
              }
            } else {
              if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
                  result[0] += -0.001408892509749275;
                } else {
                  result[0] += 0.17402519837064132;
                }
              } else {
                result[0] += -0.05194595455670514;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.029068946838379794) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
        result[0] += -0.0017398240725690095;
      } else {
        result[0] += 0.012279824877087056;
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.182021141052246982) ) ) {
        result[0] += 0.0010748118676456493;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
            result[0] += -0.083324612090717;
          } else {
            result[0] += 0.015477729600717728;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.56941866874694913) ) ) {
            result[0] += -0.015111572645818798;
          } else {
            result[0] += 0.019868036444462705;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += 0.06916725244034704;
        } else {
          result[0] += -0.009837020254335435;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.285887241363526279) ) ) {
          result[0] += 0.0004287268542202121;
        } else {
          result[0] += -0.007785522308407285;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.08626589682305917;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.012094675867556068;
                  } else {
                    result[0] += 0.0014781012909421741;
                  }
                } else {
                  result[0] += -0.0035381516868072866;
                }
              } else {
                result[0] += -0.0076009268541284425;
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0019321337445520924;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += 0.034062894815287674;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += -0.06026572782463283;
                        } else {
                          result[0] += -0.025763291204140056;
                        }
                      } else {
                        result[0] += -0.015869558598845566;
                      }
                    }
                  } else {
                    result[0] += -0.008307617747047216;
                  }
                }
              } else {
                result[0] += 0.006041833951411162;
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                  result[0] += 0.020095617460652398;
                } else {
                  result[0] += -0.014950248094305833;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.017797946929933417) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.04618721449995455;
                  } else {
                    result[0] += 0.0010788136359524817;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                      result[0] += -0.032593164755404073;
                    } else {
                      result[0] += 0.02939802806661875;
                    }
                  } else {
                    result[0] += -0.0010609305931163226;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.30779838562011896) ) ) {
                result[0] += -0.022783859524463487;
              } else {
                result[0] += 0.0339792966687718;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += -0.03880151207702988;
          } else {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.918272972106934482) ) ) {
                  result[0] += -0.005443378319300939;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                      result[0] += 0.07037600300471804;
                    } else {
                      result[0] += -0.05084661561362787;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.287653446197511542) ) ) {
                      result[0] += -0.02452340922970959;
                    } else {
                      result[0] += 0.08543478410956033;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                        result[0] += 0.043541779265467165;
                      } else {
                        result[0] += -0.07039602524690354;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.03144781712803065;
                        } else {
                          result[0] += -0.09388465859242276;
                        }
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.004309988921459562;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.778982400894165927) ) ) {
                            result[0] += -0.030886620113038967;
                          } else {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.32398128509521662) ) ) {
                              result[0] += -0.008444772272096064;
                            } else {
                              result[0] += 0.0674450700661096;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.039791968412362884;
                        } else {
                          result[0] += -0.005577206006908013;
                        }
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.002074356115696661;
                        } else {
                          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.016753425215421416;
                          } else {
                            result[0] += -0.08735512816615915;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.04702330041256212;
                          } else {
                            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                                result[0] += 0.0022223260025797664;
                              } else {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    result[0] += -0.00431864708463777;
                                  } else {
                                    result[0] += -0.04775197894333644;
                                  }
                                } else {
                                  result[0] += -0.06359386624138332;
                                }
                              }
                            } else {
                              result[0] += 0.0006328525678588406;
                            }
                          }
                        } else {
                          result[0] += 0.009414831022580046;
                        }
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.016333475809739477;
                        } else {
                          result[0] += -0.024454095174746265;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.07113513025799871;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.531673669815064365) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.868834793567657693) ) ) {
                    result[0] += 0.0201205363375926;
                  } else {
                    result[0] += -0.00013114819886305651;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.002314006923356218;
                    } else {
                      result[0] += -0.02070730552870659;
                    }
                  } else {
                    result[0] += -0.026655101202721465;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.11457478119967378;
                  } else {
                    result[0] += 0.01796021767909556;
                  }
                } else {
                  result[0] += 0.004674336122237732;
                }
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.0007013949422884631;
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.784468173980714667) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.802696108818054643) ) ) {
                  result[0] += 0.03605331784537942;
                } else {
                  result[0] += -0.02453869203613113;
                }
              } else {
                result[0] += 0.011949240679579996;
              }
            } else {
              result[0] += -0.004279953481538009;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                  result[0] += 0.015047282212626642;
                } else {
                  result[0] += -0.001427111600159718;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                  result[0] += -0.16177010396534516;
                } else {
                  result[0] += -0.018282353175036164;
                }
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                    result[0] += -0.04939635585759769;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.03507917440080314;
                    } else {
                      result[0] += -0.017723936653669937;
                    }
                  }
                } else {
                  result[0] += 0.025229264636124737;
                }
              } else {
                result[0] += 0.0006823258718138024;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
            result[0] += 0.014901309573463285;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.001773655723399108;
              } else {
                result[0] += -0.014458798028975121;
              }
            } else {
              result[0] += -0.01837789043296709;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.341600894927979404) ) ) {
          result[0] += -0.06495775720670537;
        } else {
          result[0] += -0.01118382202213101;
        }
      }
    } else {
      result[0] += 0.0001837201588672657;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.662244915962219682) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += 0.09071325034534478;
        } else {
          result[0] += -0.042422820731242906;
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.617236852645874912) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.16688237440433673;
                } else {
                  result[0] += 0.045546837713092286;
                }
              } else {
                result[0] += -0.06342389930286003;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                  result[0] += -0.04292298698962526;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.094205617904663974) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                      result[0] += -0.07966951995516035;
                    } else {
                      result[0] += 0.05054868577957954;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                      result[0] += 0.12047221922528553;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.350240230560303178) ) ) {
                          result[0] += 0.02231026035655132;
                        } else {
                          result[0] += -0.01601201448192594;
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.01305858076430999;
                        } else {
                          result[0] += 0.04435723771595301;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.022335333412736493;
                    } else {
                      result[0] += 0.017587931795953634;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.007753909137517379;
                    } else {
                      result[0] += 0.05757344968297732;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                    result[0] += 0.13143651124607325;
                  } else {
                    result[0] += -0.03590887205272147;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.138333082199097124) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.06330776748973792;
              } else {
                result[0] += 0.10880234097712796;
              }
            } else {
              result[0] += -0.06803806554373151;
            }
          }
        } else {
          if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.040527637376436824;
          } else {
            result[0] += 0.15812060520707522;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.22084951400757014) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.17063544628365462;
              } else {
                result[0] += -0.029572311960533573;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                  result[0] += 0.005641084765507007;
                } else {
                  result[0] += 0.05194263451441871;
                }
              } else {
                result[0] += -0.026533382927101907;
              }
            }
          } else {
            result[0] += -0.0752732106708623;
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.08487257093834254;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                    result[0] += -0.05799383141354769;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                      result[0] += -0.028031404747859947;
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.016185762015045995;
                      } else {
                        result[0] += 0.060091164149644695;
                      }
                    }
                  }
                } else {
                  result[0] += -0.06624690619215944;
                }
              } else {
                result[0] += -0.04470209957707169;
              }
            }
          } else {
            result[0] += -0.06484410094055966;
          }
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += -0.0003749675013072755;
              } else {
                result[0] += 0.0887356969594745;
              }
            } else {
              result[0] += 0.12777787046410263;
            }
          } else {
            result[0] += -0.015352673170259538;
          }
        } else {
          result[0] += -0.014753793268288793;
        }
      }
    }
  }
  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += -0.0005087392375572977;
    } else {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.10649580357910789;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.10427457835686771;
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.11500127673519381;
                } else {
                  result[0] += 0.07420230808977356;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
              result[0] += 0.09836290139983822;
            } else {
              result[0] += -0.006050308410448443;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.051747083663941318) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                  result[0] += -0.026558052722623467;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.347943067550660068) ) ) {
                    result[0] += 0.12069437774751252;
                  } else {
                    result[0] += 0.018324102613780226;
                  }
                }
              } else {
                result[0] += -0.037052000474503084;
              }
            } else {
              result[0] += -0.06965577049219776;
            }
          } else {
            result[0] += 0.003633322359236025;
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.513969182968140537) ) ) {
                  result[0] += 0.011523979641978939;
                } else {
                  result[0] += 0.0005162566475278027;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
                  result[0] += -0.12566607741910457;
                } else {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.287653446197511542) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.029997882045684106;
                        } else {
                          result[0] += 0.02331335177349098;
                        }
                      } else {
                        result[0] += 0.03421572047732149;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.35306882858276456) ) ) {
                        result[0] += -0.0060030092048350275;
                      } else {
                        result[0] += 0.04554280689819834;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.010094406257480404;
                    } else {
                      result[0] += -0.0009574437668668384;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01696564173266076;
                } else {
                  result[0] += -0.000398754747942419;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.0004558226633397682;
                } else {
                  result[0] += -0.08343823801946226;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0028739684147024783;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.13318983878443175;
                    } else {
                      result[0] += 0.008244063237564457;
                    }
                  } else {
                    result[0] += 0.004214172930061779;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                      result[0] += -0.04006286755960865;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.013292589185456448;
                        } else {
                          result[0] += 0.11695382714994679;
                        }
                      } else {
                        result[0] += -0.02843354135272044;
                      }
                    }
                  } else {
                    result[0] += -0.05049334915038343;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                  result[0] += 0.005848256562588438;
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.567899227142334428) ) ) {
                    result[0] += 0.006546164063232858;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.397998809814454013) ) ) {
                      result[0] += -0.01349324551400648;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.855006217956543857) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.19447340084260825;
                          } else {
                            result[0] += -0.033709483928184646;
                          }
                        } else {
                          result[0] += -0.031013849194523782;
                        }
                      } else {
                        result[0] += -0.022468788537796894;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.464467763900757724) ) ) {
                  result[0] += 0.00439524004625375;
                } else {
                  result[0] += -0.01598848304000987;
                }
              }
            }
          }
        } else {
          result[0] += 0.003979393107412787;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.655405282974244052) ) ) {
      if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
        result[0] += -0.08954675527204373;
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.917405366897583452) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
              result[0] += -0.040134819377064264;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.802696108818054643) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.1324146669559558;
                } else {
                  result[0] += 0.03651847931191392;
                }
              } else {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.750972747802735263) ) ) {
                    result[0] += 0.029365381908204577;
                  } else {
                    result[0] += -0.0066313683337632994;
                  }
                } else {
                  result[0] += -0.01022079922130769;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.07457304427325231;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.010721196344124898;
              } else {
                result[0] += -0.035545848604835666;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.249904870986938921) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 1.208825122229855;
                  } else {
                    result[0] += 0.23989331525959848;
                  }
                } else {
                  result[0] += 0.021084635234157715;
                }
              } else {
                result[0] += -0.07492006769115228;
              }
            } else {
              result[0] += -0.056481841795683;
            }
          } else {
            result[0] += 0.1090954253966206;
          }
        }
      }
    } else {
      result[0] += 0.09686769261449213;
    }
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.018260965575418616;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
            result[0] += -0.007575166629491097;
          } else {
            result[0] += -0.03830803371012455;
          }
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.487163543701172763) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
              result[0] += -0.008148674952868895;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                    result[0] += 0.0276546125832778;
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.138731002807618076) ) ) {
                      result[0] += 0.01982520213087817;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.329314231872559482) ) ) {
                        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.011326246541691024;
                        } else {
                          result[0] += -0.06346155227905577;
                        }
                      } else {
                        result[0] += -0.015770637954082095;
                      }
                    }
                  }
                } else {
                  result[0] += 0.004515003211785848;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.18441450692713612;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.006200002104717832;
                      } else {
                        result[0] += 0.026500894112599385;
                      }
                    } else {
                      result[0] += -0.08415804710967328;
                    }
                  }
                } else {
                  result[0] += -0.020686408623593586;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.046203896634472386;
            } else {
              result[0] += -0.016916316356180926;
            }
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.026652704475012313;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.04534961645098898;
                  } else {
                    result[0] += 0.05096051843696006;
                  }
                } else {
                  result[0] += -0.034385205655691406;
                }
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.1124487496165165;
              } else {
                result[0] += -0.005633625662335817;
              }
            }
          } else {
            result[0] += 0.00444801032249759;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.071567356586456743) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                result[0] += -0.013499467730123444;
              } else {
                result[0] += -0.16465624993964817;
              }
            } else {
              result[0] += -0.23499844436088646;
            }
          } else {
            result[0] += 0.0005624177678533736;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                result[0] += 0.07548966551927788;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += -0.14408885681241382;
                } else {
                  result[0] += -0.005585162460578646;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
                result[0] += 0.04366875893255806;
              } else {
                result[0] += 0.1488376723792816;
              }
            }
          } else {
            result[0] += 0.16359714714354232;
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.778982400894165927) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                      result[0] += 0.08358908528713715;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.041303525843154554;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.025192260742188388) ) ) {
                          result[0] += -0.021691482364984988;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                            result[0] += -0.1198005010196487;
                          } else {
                            result[0] += 0.006041750324330328;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.938867926597595659) ) ) {
                      result[0] += -0.02206659371576917;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                        result[0] += 0.008172682716511196;
                      } else {
                        result[0] += 0.04206532468796857;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0266221649814764;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                  result[0] += -0.047522105277847476;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.10135770351130756;
                  } else {
                    result[0] += -0.012724042585466984;
                  }
                }
              }
            } else {
              result[0] += 0.01140893719874233;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.105651378631592685) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += -0.12611158618702173;
              } else {
                result[0] += -0.0301409541994367;
              }
            } else {
              if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                  result[0] += 0.04221131709667888;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += 0.017238773601465265;
                      } else {
                        result[0] += -0.04873378545229395;
                      }
                    } else {
                      result[0] += 0.009828659141154602;
                    }
                  } else {
                    result[0] += -0.05030343421263195;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.0002634174999926405;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.799905776977539951) ) ) {
                        result[0] += 0.0026339458196833958;
                      } else {
                        result[0] += -0.015936915077653168;
                      }
                    } else {
                      result[0] += -0.018160055777599916;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03147524481648722;
                  } else {
                    result[0] += 0.0003427473634670511;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.341600894927979404) ) ) {
            result[0] += -0.0721102709324286;
          } else {
            result[0] += -0.017133659410235143;
          }
        }
      }
    }
  } else {
    result[0] += 8.103595220351084e-05;
  }
  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.497866153717041238) ) ) {
      result[0] += 0.030099512455889706;
    } else {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53326439857482999) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54220247268676935) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.013907881587069435;
            } else {
              result[0] += 0.0695965516079402;
            }
          } else {
            result[0] += 0.02826228141723008;
          }
        } else {
          result[0] += 0.03693156335340773;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                result[0] += 0.0032245175558568728;
              } else {
                result[0] += -0.02809606723774973;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.214365959167481357) ) ) {
                result[0] += 0.007367697027486933;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.426736354827881748) ) ) {
                      result[0] += 0.0014015316347794128;
                    } else {
                      result[0] += -0.03351021652590908;
                    }
                  } else {
                    result[0] += 0.031970841404843686;
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.06713625893559079;
                  } else {
                    result[0] += -0.003940777949731284;
                  }
                }
              }
            }
          } else {
            result[0] += -0.004930529392907643;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.0033185985283134497;
          } else {
            result[0] += -0.012475566060068388;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
      result[0] += -0.013865529267825786;
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.29667711257934748) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.384830474853516513) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.042238902890196495;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                  result[0] += -0.07331836311906008;
                } else {
                  result[0] += 0.03436532644545336;
                }
              }
            } else {
              result[0] += -0.021675143293463418;
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
              result[0] += 0.08598400838717155;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.2531323432922381) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.241523027420044833) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.662244915962219682) ) ) {
                      result[0] += -0.09491668840102163;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                        result[0] += -0.05317615620810495;
                      } else {
                        result[0] += 0.08434062475832499;
                      }
                    }
                  } else {
                    result[0] += -0.11332328596219553;
                  }
                } else {
                  result[0] += 0.06712135191280928;
                }
              } else {
                result[0] += 0.09061368807268311;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
            result[0] += 0.08481470597767625;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
              result[0] += -0.0295434888514784;
            } else {
              result[0] += 0.0034704998912973026;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.07759690377969275;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.510617971420288974) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.0012350748387757345;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.643222332000734198) ) ) {
                    result[0] += -0.1280455969066828;
                  } else {
                    result[0] += 0.03651136457287059;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                  result[0] += -0.015971893303447053;
                } else {
                  result[0] += -0.07385332634611017;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.001872670723141008;
              } else {
                result[0] += -0.04084864157227977;
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.005295767497348756;
              } else {
                result[0] += 0.02414432992438685;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09806728363037287) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.000695018571583824;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                      result[0] += -0.00870846597051722;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                        result[0] += 0.008277274888314352;
                      } else {
                        result[0] += 0.06038439767757281;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.02694531678160289;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.11139068183689123;
                      } else {
                        result[0] += 0.012817450743118334;
                      }
                    } else {
                      result[0] += 0.005870232857459471;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.0001405386198723016;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.004919120838472359;
                    } else {
                      result[0] += -0.06843560330267391;
                    }
                  } else {
                    result[0] += -0.03434206476445609;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.467917680740357333) ) ) {
                  result[0] += 0.004981651700486678;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.005187365950315698;
                    } else {
                      result[0] += -0.04196699159966685;
                    }
                  } else {
                    result[0] += -0.00010620767701997564;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.321723937988282138) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.95386886596679865) ) ) {
                    result[0] += 0.007492217011283598;
                  } else {
                    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.020400036430578458;
                    } else {
                      result[0] += 0.03104125501913035;
                    }
                  }
                } else {
                  result[0] += 0.03552246532991313;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.197461366653443271) ) ) {
              result[0] += -0.00034073937891316837;
            } else {
              result[0] += -0.01334690506838334;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      result[0] += -0.002214391016140714;
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.005894113199447364;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.0017038033548003664;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.91907978057861506) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.00853296792756005;
                  } else {
                    result[0] += -0.03620502094086694;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.009106033506336935;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                      result[0] += -0.01512853262117066;
                    } else {
                      result[0] += 0.04236047533812792;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += 0.007001319143206299;
          }
        } else {
          result[0] += -0.030606498116145117;
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                result[0] += -0.002290679708926526;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.038699583826424466;
                } else {
                  result[0] += 0.003380682096282599;
                }
              }
            } else {
              result[0] += 0.009094072131865364;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
              result[0] += 0.0017919655234785272;
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.011799450176401358;
              } else {
                result[0] += 0.05021583758535077;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                result[0] += -0.0020559898768889966;
              } else {
                result[0] += 0.01565090050421104;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.637949228286744052) ) ) {
                  result[0] += 0.0663643094654539;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                      result[0] += 0.03238632463816685;
                    } else {
                      result[0] += -0.12786033708624492;
                    }
                  } else {
                    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += -0.0017910373490160178;
                    } else {
                      result[0] += 0.07694534555875744;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.08634819538999988;
                } else {
                  result[0] += -0.06784870802466919;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.01708412170410334) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.0023933876993776134;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.014788627624512607) ) ) {
                    result[0] += -0.029803994216467674;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.05333482296461715;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.572941064834595615) ) ) {
                        result[0] += -0.041799703352758406;
                      } else {
                        result[0] += 0.04566633004905568;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.010148536918117299;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
                              result[0] += 0.01969206409888091;
                            } else {
                              result[0] += -0.07161317379369243;
                            }
                          } else {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                              result[0] += 0.1453222716974084;
                            } else {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                                result[0] += -0.06391579700540981;
                              } else {
                                result[0] += 0.02162105230699779;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                                result[0] += -0.050054092347400506;
                              } else {
                                result[0] += 0.022780384580378125;
                              }
                            } else {
                              result[0] += -0.08357351119247503;
                            }
                          } else {
                            result[0] += 0.015461508108959167;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
                              result[0] += 0.05352487300048922;
                            } else {
                              result[0] += -0.06419168298759491;
                            }
                          } else {
                            result[0] += 0.049196673446771436;
                          }
                        } else {
                          result[0] += 0.02070594956691549;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                        result[0] += 0.008681593793864146;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.379217386245728427) ) ) {
                          result[0] += 0.014728791950162384;
                        } else {
                          result[0] += -0.09208834329394845;
                        }
                      }
                    } else {
                      result[0] += 0.01987476531093455;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.158952236175537998) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.95386886596679865) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.153024196624756748) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.342454433441162998) ) ) {
                            result[0] += 0.007622530651999297;
                          } else {
                            result[0] += -0.07056741807703613;
                          }
                        } else {
                          result[0] += 0.006362919112887862;
                        }
                      } else {
                        result[0] += 0.055151382311794866;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.518026351928711826) ) ) {
                        result[0] += 0.03246349606837813;
                      } else {
                        result[0] += 0.12998493800127953;
                      }
                    }
                  } else {
                    result[0] += 0.0729759370558176;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
                result[0] += -0.005474018703703606;
              } else {
                result[0] += -0.11515309193553458;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
      result[0] += 0.0001273891143903363;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.591613531112671787) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.855006217956543857) ) ) {
            result[0] += -0.003928451502788561;
          } else {
            result[0] += 0.007012690680077125;
          }
        } else {
          result[0] += -0.014780370953866688;
        }
      } else {
        result[0] += -0.02319289693811241;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.084203958511353427) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.006792289306424325;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.004425555163654296;
              } else {
                result[0] += 0.05563790621262687;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.007970047393183972;
            } else {
              result[0] += -0.027138628773256446;
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.010086006285668372;
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.600145101547242099) ) ) {
                result[0] += -0.0029696424228206914;
              } else {
                result[0] += -0.04889554003083879;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.877672910690308505) ) ) {
              result[0] += -0.003412265150111533;
            } else {
              result[0] += -0.024757549141625177;
            }
          }
        }
      } else {
        result[0] += 0.001835016184272893;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.003838300704956943) ) ) {
                  result[0] += 0.0023875095419651376;
                } else {
                  result[0] += -0.035893984885984935;
                }
              } else {
                result[0] += 0.01589391050289924;
              }
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.15227253556577447;
                    } else {
                      result[0] += -0.005196943827617665;
                    }
                  } else {
                    result[0] += -0.021090115465435908;
                  }
                } else {
                  result[0] += -0.04828975135253033;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
                  result[0] += -0.07770150441768292;
                } else {
                  if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0018515206304378804;
                  } else {
                    result[0] += 0.024567361222318465;
                  }
                }
              }
            }
          } else {
            result[0] += 0.006438984307705353;
          }
        } else {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.04493507401015841;
          } else {
            result[0] += -0.007390065853991258;
          }
        }
      } else {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.515218973159790483) ) ) {
                result[0] += -0.0017992354626197049;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.10022124820837791;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.381086945533752885) ) ) {
                        result[0] += -0.020737676789105985;
                      } else {
                        result[0] += 0.08298118236108688;
                      }
                    }
                  } else {
                    result[0] += 0.0006116289573613509;
                  }
                } else {
                  result[0] += 0.003777387745856422;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.467917680740357333) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                    result[0] += 0.009973214579250024;
                  } else {
                    result[0] += 0.08992825182311455;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.778982400894165927) ) ) {
                    result[0] += 0.024077929873658227;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08693766869283574;
                    } else {
                      result[0] += -0.01901576942654923;
                    }
                  }
                }
              } else {
                result[0] += 0.011592242216017956;
              }
            }
          } else {
            result[0] += -0.006268494367546587;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.011618877356714496;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82428741455078303) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -5.88241923796273e-05;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                        result[0] += -0.02935207168799436;
                      } else {
                        result[0] += 0.019065780128754553;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += 0.029585664830935405;
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.0062029135231363025;
                      } else {
                        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.07723487630873024;
                        } else {
                          result[0] += -0.016569675724882885;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.02517163785961243;
                    } else {
                      result[0] += -0.0148054387818881;
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                      result[0] += 0.032414244768412925;
                    } else {
                      result[0] += -0.04392626148511051;
                    }
                  }
                }
              } else {
                result[0] += 0.010239795968243124;
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.088880300521851474) ) ) {
                  result[0] += 0.0015093631301230316;
                } else {
                  result[0] += -0.05409208159192428;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                      result[0] += -0.03755908659785709;
                    } else {
                      result[0] += 0.052697468025400464;
                    }
                  } else {
                    result[0] += -0.0586130266046061;
                  }
                } else {
                  result[0] += 0.015644512028431756;
                }
              }
            } else {
              result[0] += 0.014655872098302694;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.24173307418823331) ) ) {
        if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += 0.02641485394865923;
        } else {
          result[0] += -0.010512006269661431;
        }
      } else {
        if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
          result[0] += -0.009327060523138581;
        } else {
          result[0] += -0.05924646440930938;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
        result[0] += 0.000550563216489342;
      } else {
        result[0] += -0.0030949934905707124;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.00028821229259825903;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
          result[0] += -0.05572115870129062;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
            result[0] += 0.02742187415358587;
          } else {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.004916995810266042;
            } else {
              result[0] += -0.027293775168077218;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.005292678426872154;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
                  result[0] += -0.008789526064245417;
                } else {
                  result[0] += -0.04201515543815406;
                }
              } else {
                if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.002289443083578586;
                } else {
                  result[0] += 0.022531088373404822;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += -0.0013073908712159812;
              } else {
                result[0] += -0.04315548011496723;
              }
            } else {
              result[0] += 0.008537292986482882;
            }
          }
        } else {
          result[0] += -0.023933110527108746;
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.450390577316285068) ) ) {
                    result[0] += 0.00474346492594872;
                  } else {
                    result[0] += 0.05948568344645309;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.843275547027588779) ) ) {
                    result[0] += 0.0048569754424840845;
                  } else {
                    result[0] += -0.004894485183140495;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.205624103546144354) ) ) {
                  result[0] += 0.0031565690559028597;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.032524101395811224;
                    } else {
                      result[0] += -0.010790161526976566;
                    }
                  } else {
                    result[0] += 0.08107684952483843;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                result[0] += 0.021717932191386385;
              } else {
                result[0] += -0.00015414046414666645;
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.071567356586456743) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                  result[0] += 0.022394195782151634;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0011886582862494657;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.011726453966936326;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
                        result[0] += 0.11486503507031853;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.55753517150879084) ) ) {
                          result[0] += -0.07932203875370988;
                        } else {
                          result[0] += 0.10526458737982541;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.033414087647233685;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                result[0] += -0.0029502103964807805;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.09435406029154803;
                  } else {
                    result[0] += 0.04432879047779351;
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.551017761230469638) ) ) {
                    result[0] += -0.043272621532099195;
                  } else {
                    result[0] += -0.2768107348049008;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.05931575939835374;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.825422286987305576) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.0049068014810218115;
                  } else {
                    result[0] += -0.044859888614775384;
                  }
                } else {
                  result[0] += 0.002033842846004591;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.431901693344116655) ) ) {
                    result[0] += -0.06736132746595595;
                  } else {
                    result[0] += 0.017179602578768393;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.008179404620249997;
                    } else {
                      result[0] += 0.02837679736298797;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.136462926864624912) ) ) {
                      result[0] += -0.0005279530225452162;
                    } else {
                      result[0] += 0.016879142904011404;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.010534427993700794;
                } else {
                  result[0] += -0.09662638780947333;
                }
              } else {
                result[0] += -0.01982469553151563;
              }
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.015284108175459997;
              } else {
                result[0] += 0.08020450078228919;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += 0.0007268905037765052;
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
            result[0] += 0.009498072174369286;
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.008133399419148429;
            } else {
              result[0] += -0.02163861922420483;
            }
          }
        } else {
          result[0] += -0.03770613233249864;
        }
      }
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.255632162094117099) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
            result[0] += 0.0009697364603403808;
          } else {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.005678130860568722;
            } else {
              result[0] += -0.029933832497009996;
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.005064934260887503;
            } else {
              result[0] += -0.030742507565395056;
            }
          } else {
            result[0] += -0.022232227438278424;
          }
        }
      } else {
        result[0] += 0.004672553623742763;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
        result[0] += -0.0010365014788022372;
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.008889142616990538;
          } else {
            result[0] += -0.035224020742808865;
          }
        } else {
          result[0] += 0.020228185783243098;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
              result[0] += -0.011476648717407499;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.0020151202914402977;
                } else {
                  result[0] += 0.032682681421112574;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0026586474074021117;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.04932466470040568;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.02299397234690765;
                      } else {
                        result[0] += 0.014640353809431988;
                      }
                    }
                  } else {
                    result[0] += -0.004888493754641746;
                  }
                }
              }
            }
          } else {
            result[0] += 0.005456815579760271;
          }
        } else {
          result[0] += -0.020455737227645324;
        }
      } else {
        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
              result[0] += -0.002252323503613198;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.00022700150195404864;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.012514310223844966;
                  } else {
                    result[0] += -0.06413308581833309;
                  }
                }
              } else {
                result[0] += -0.05849655615635049;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                result[0] += -0.003798794490028721;
              } else {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.02188759676771186;
                } else {
                  result[0] += -0.016210544908787105;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.041921615600587714) ) ) {
                result[0] += 0.023745147970669997;
              } else {
                result[0] += 0.005180135208714672;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.634540319442749912) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                result[0] += 0.025521081925938046;
              } else {
                result[0] += -0.05954417186339235;
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.07733613949216776;
                } else {
                  result[0] += 0.009751612510377284;
                }
              } else {
                result[0] += -0.011581600598471301;
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.48298668861389249) ) ) {
              result[0] += 0.0038671254802402095;
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.10455857155303137;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                        result[0] += 0.016823885385151626;
                      } else {
                        result[0] += -0.03217888681280267;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.01578349734475766;
                      } else {
                        result[0] += 0.04056731301589998;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.01522288128592282;
                    } else {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.01124029957473717;
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += 0.0651644653020829;
                        } else {
                          result[0] += -0.009645138349697643;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.03094757355554655;
                  }
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.01866909769528482;
                } else {
                  result[0] += 0.07775288843900774;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.357691764831543413) ) ) {
        result[0] += -0.010033835780181488;
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.553712725639343706) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.124530076980591708) ) ) {
            result[0] += 0.0011502884320486538;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.40000796318054288) ) ) {
              result[0] += 0.0006693918044294945;
            } else {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.012338317901618217;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.02269606500536621;
                  } else {
                    result[0] += -0.0012001311796996606;
                  }
                } else {
                  result[0] += 0.0027820707240936837;
                }
              }
            }
          }
        } else {
          result[0] += -0.01886937645907606;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.02087272291546844;
        } else {
          result[0] += -0.03081671029145196;
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.014678420525553699;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.909855604171753818) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.745876312255860263) ) ) {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.022935630566629485;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                        result[0] += 0.11894645899502104;
                      } else {
                        result[0] += 0.0258908529007728;
                      }
                    } else {
                      result[0] += -0.022237248125844263;
                    }
                  }
                } else {
                  result[0] += -0.04478035459647628;
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.030119491843803217;
                } else {
                  result[0] += 0.073425919599586;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                result[0] += 0.07330071002525405;
              } else {
                result[0] += -0.03147707482719471;
              }
            }
          }
        } else {
          result[0] += -0.039842426443679205;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.534971714019776279) ) ) {
      result[0] += 0.0041781561517980235;
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.112574815750122958) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += 0.03931292567299002;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
              result[0] += -0.04122642741423089;
            } else {
              result[0] += 0.01280000694279087;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.14828208848833868;
            } else {
              result[0] += -0.010550075202580032;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.020029200767704966;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.216319084167481357) ) ) {
                result[0] += -0.04999858275643154;
              } else {
                result[0] += 0.00949149045328888;
              }
            }
          }
        }
      } else {
        result[0] += 0.00631201509263692;
      }
    }
  } else {
    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.634540319442749912) ) ) {
              result[0] += 0.005837511785833887;
            } else {
              if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.06948820913936489;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.764287948608400214) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.924581527709961826) ) ) {
                      result[0] += 0.026207081227906548;
                    } else {
                      result[0] += -0.04563854199121829;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.013359404327913858;
                    } else {
                      result[0] += -0.10161115843615812;
                    }
                  }
                } else {
                  result[0] += -0.0054870840723125905;
                }
              }
            }
          } else {
            result[0] += -0.10175037567717145;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.094205617904663974) ) ) {
              result[0] += 0.0011542602842552607;
            } else {
              result[0] += 0.03361151629820865;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.680161952972413886) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.285887241363526279) ) ) {
                    result[0] += 0.0370315868851652;
                  } else {
                    result[0] += 0.011967047119907839;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.080866575241089755) ) ) {
                      result[0] += 0.0011400477371784898;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.01827298361416353;
                      } else {
                        result[0] += 0.1553640149490292;
                      }
                    }
                  } else {
                    result[0] += -0.03153991878838629;
                  }
                }
              } else {
                result[0] += -0.003551724056611435;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.709793567657472479) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.772694945335388628) ) ) {
                  result[0] += -0.1713065550301086;
                } else {
                  result[0] += -0.006212787661399678;
                }
              } else {
                result[0] += -0.011823720430825304;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.12546978894731634;
          } else {
            result[0] += -0.014910383187626991;
          }
        } else {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                  result[0] += -0.025962600717028935;
                } else {
                  result[0] += -0.07702922654330735;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
                    result[0] += -0.011076107403414154;
                  } else {
                    result[0] += -0.05181870016939662;
                  }
                } else {
                  result[0] += -0.044558397472718045;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.835998296737671787) ) ) {
                  if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.58491539955139249) ) ) {
                      if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.14328689165960107;
                      } else {
                        result[0] += 0.01880683818450446;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                        result[0] += 0.13414572423702154;
                      } else {
                        result[0] += -0.01045032374684067;
                      }
                    }
                  } else {
                    result[0] += -0.11480328986781015;
                  }
                } else {
                  result[0] += 0.038773516071785384;
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.025406354113011523;
                } else {
                  result[0] += -0.04794129119212792;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.501502752304078037) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.0055509834327168136;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.154959201812744585) ) ) {
                      result[0] += 0.02761011663862094;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.05114537632725709;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.941167116165162021) ) ) {
                          result[0] += 0.1073255343842732;
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.047023847126114376;
                          } else {
                            result[0] += 0.021391322967844975;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.219419956207276279) ) ) {
                        result[0] += 0.06698921382811014;
                      } else {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                          result[0] += -0.10755190504409683;
                        } else {
                          result[0] += 0.005447849538789105;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += 0.06196165612202206;
                      } else {
                        result[0] += 0.1476216276287092;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0352796874254066;
                    } else {
                      result[0] += 0.028442747613846034;
                    }
                  }
                }
              } else {
                result[0] += 0.1837856118141205;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.08629430220969564;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.9569747039090549;
                } else {
                  result[0] += -0.02309861277294555;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += 6.570088593536666e-05;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += 0.00025855977513280034;
    } else {
      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.04637106519471206;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.00857270186867885;
                } else {
                  result[0] += 0.029224542691468444;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)46.00000000000000711) ) ) {
                  result[0] += 0.08161558377651199;
                } else {
                  result[0] += -0.012560236954204643;
                }
              }
            } else {
              result[0] += -0.006823522616533808;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.05591979474704604;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.844227671623230425) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.431901693344116655) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.015957062974392183;
                    } else {
                      result[0] += -0.012890780420788823;
                    }
                  } else {
                    result[0] += -0.14867452278638332;
                  }
                } else {
                  result[0] += -0.007534109905112678;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51693725585937678) ) ) {
                    result[0] += 0.07043566450250255;
                  } else {
                    result[0] += -0.03296961290907807;
                  }
                } else {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815814018249513495) ) ) {
                      result[0] += -0.03412618863031311;
                    } else {
                      result[0] += 0.11426235392480055;
                    }
                  } else {
                    result[0] += -0.04609673005840423;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += -0.024209230662344462;
            } else {
              result[0] += -0.006650275529088575;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.693369150161744052) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.553712725639343706) ) ) {
              result[0] += 0.006751416675634563;
            } else {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.0055201163025076555;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.042661172597600695;
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                    result[0] += -0.015279356890931677;
                  } else {
                    result[0] += 0.04213017713752583;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                  result[0] += -0.0034054138436817117;
                } else {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    result[0] += -0.03523706714934241;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                      result[0] += -0.054170424557181164;
                    } else {
                      result[0] += 0.03552742081056747;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.531007289886475498) ) ) {
                      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.505334615707398349) ) ) {
                          result[0] += 0.023716206806332796;
                        } else {
                          result[0] += -0.00011606475901200058;
                        }
                      } else {
                        result[0] += 0.0011989726641180617;
                      }
                    } else {
                      result[0] += 0.03936373102952142;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.600145101547242099) ) ) {
                      result[0] += -0.03219132223678511;
                    } else {
                      result[0] += 0.03693176141630274;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                    result[0] += 0.03785049947504002;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
                        result[0] += 0.003029023263778028;
                      } else {
                        result[0] += 0.1191456342431956;
                      }
                    } else {
                      result[0] += -0.03784535433263807;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.10121199599204292;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
              result[0] += 0.04566231682290131;
            } else {
              result[0] += -0.03498501806345526;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.816582441329956943) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.19447278976440607) ) ) {
                    result[0] += 0.003160143044109878;
                  } else {
                    result[0] += 0.035453880782007174;
                  }
                } else {
                  result[0] += 0.04898480872969259;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.53326439857482999) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.055214703255579284;
                        } else {
                          result[0] += -0.000685874405593191;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.342454433441162998) ) ) {
                          result[0] += -0.08431303770907576;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                            result[0] += 0.043095404939577225;
                          } else {
                            result[0] += -0.11193442376671699;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.87502956390381037) ) ) {
                          result[0] += 0.0076711914920749085;
                        } else {
                          result[0] += 0.050733610970065915;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.262283086776734287) ) ) {
                          result[0] += 0.02471093985939862;
                        } else {
                          result[0] += 0.07612727805978398;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.918272972106934482) ) ) {
                      result[0] += -0.05117701411273326;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                        result[0] += -0.031224344891288548;
                      } else {
                        result[0] += 0.10530430268111904;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.038446028505768524;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.05141562464318522;
                    } else {
                      result[0] += -0.10358517693202998;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.03668799250232535;
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        result[0] += 0.03772739709804828;
      } else {
        result[0] += 0.003321004257626655;
      }
    } else {
      result[0] += 9.952413212207933e-05;
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.00043672889541340493;
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        result[0] += 0.03858921929095131;
      } else {
        result[0] += 0.003217871066209065;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.445957899093628818) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += -0.020638322951561774;
                } else {
                  result[0] += 0.08454550568713018;
                }
              } else {
                result[0] += 0.03721088560071034;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.433569431304932529) ) ) {
                result[0] += 0.007516935040709272;
              } else {
                result[0] += 0.030199720852891906;
              }
            }
          } else {
            if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.784468173980714667) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.06552552365447038;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.963667154312134677) ) ) {
                      result[0] += -0.0034194442316475924;
                    } else {
                      result[0] += 0.089405929507995;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.02206822675864033;
                  } else {
                    result[0] += 0.016229988590661437;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0960660165992665;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.737386107444763628) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                        result[0] += 0.07379617264742164;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.381086945533752885) ) ) {
                          result[0] += -0.10727672142150013;
                        } else {
                          result[0] += 0.010310662608583548;
                        }
                      }
                    } else {
                      result[0] += -0.020219457464217314;
                    }
                  } else {
                    result[0] += -0.061230063294197404;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.005126805181951643;
                } else {
                  result[0] += 0.03171666398549582;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                  result[0] += 0.039128032997827564;
                } else {
                  result[0] += -0.028197251481875203;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
            result[0] += 0.04775570528437678;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
              result[0] += -0.01725242299494471;
            } else {
              result[0] += -0.05908163104866493;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
            result[0] += 0.005380221728756636;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.534971714019776279) ) ) {
              result[0] += -0.014446410937971442;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.0055400269385129305;
              } else {
                result[0] += 0.051487686490190344;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.918272972106934482) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.547126770019532138) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.029068946838379794) ) ) {
                    if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.198252916336060458) ) ) {
                        result[0] += 0.07572532966888271;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.025459368808794777;
                        } else {
                          result[0] += -0.1281275586205137;
                        }
                      }
                    } else {
                      result[0] += -0.06085658200759778;
                    }
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.270308971405030185) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.192109584808350498) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                          result[0] += -0.016028482900770854;
                        } else {
                          result[0] += 0.08180269046238779;
                        }
                      } else {
                        result[0] += -0.06473226281369872;
                      }
                    } else {
                      result[0] += 0.009357645134282904;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.142630577087403232) ) ) {
                    result[0] += -0.04219983192085214;
                  } else {
                    result[0] += 0.010966806360480058;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.041387319564820224) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.0325811913319604;
                  } else {
                    result[0] += -0.0993544851986094;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.023565430688346795;
                  } else {
                    result[0] += 0.04074405135019287;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.628996372222901279) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.851041555404663974) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.074332714080812323) ) ) {
                      result[0] += -0.021318487663372958;
                    } else {
                      result[0] += 0.04322362403872245;
                    }
                  } else {
                    result[0] += 0.02100436897597545;
                  }
                } else {
                  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.940167903900147373) ) ) {
                      result[0] += 0.018530131903179183;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.036049604415894443) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.102759599685669833) ) ) {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                                result[0] += 0.020630916429063118;
                              } else {
                                result[0] += -0.03376428440739311;
                              }
                            } else {
                              result[0] += 0.04028923525150328;
                            }
                          } else {
                            result[0] += 0.06231418715551773;
                          }
                        } else {
                          result[0] += -0.00021936770942055468;
                        }
                      } else {
                        result[0] += -0.015080078953536727;
                      }
                    }
                  } else {
                    result[0] += -0.011471780959502107;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.06599733115909673;
                } else {
                  result[0] += -0.025185840301061704;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.161920547485352451) ) ) {
              result[0] += -0.0007265470174569092;
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.074540649601473;
                    } else {
                      result[0] += 0.006572344991345115;
                    }
                  } else {
                    result[0] += 0.03620937211625186;
                  }
                } else {
                  result[0] += 0.03581259151212559;
                }
              } else {
                result[0] += -0.00760821034452206;
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += -0.00042409626540004667;
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.830334186553955966) ) ) {
          result[0] += -0.0073372572028500855;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37109279632568537) ) ) {
            result[0] += 0.08608206826501108;
          } else {
            result[0] += -0.058602254348351195;
          }
        }
      } else {
        result[0] += 0.003103207385804602;
      }
    } else {
      if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.623839378356934482) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.311204195022583896) ) ) {
                      result[0] += -0.004557364640662209;
                    } else {
                      result[0] += -0.04182355933185025;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                      result[0] += -0.05084138669618243;
                    } else {
                      result[0] += 0.007780821696224606;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.06406347198954766;
                    } else {
                      result[0] += 0.08161371070743162;
                    }
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.00947920373114093;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.06618835234794802;
                      } else {
                        result[0] += 0.04209239634313896;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.69067406654357999) ) ) {
                  result[0] += 0.07493966350665993;
                } else {
                  result[0] += 0.009554351703441953;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.433569431304932529) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.01726991992800008;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.209340095520020419) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.01895729422916408;
                      } else {
                        result[0] += 0.06839316180894232;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                        result[0] += -0.08058713537564438;
                      } else {
                        result[0] += 0.008406896990271785;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.74845767021179288) ) ) {
                    result[0] += 0.04384567575063281;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.641084194183350498) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                        result[0] += -0.14395204259947642;
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                          result[0] += 0.02285504301539076;
                        } else {
                          result[0] += -0.014613984509410708;
                        }
                      }
                    } else {
                      result[0] += 0.040035085404416144;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.05056879447492323;
                      } else {
                        result[0] += -0.020613345375007808;
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.868494272232056552) ) ) {
                        result[0] += 0.0017689856101800437;
                      } else {
                        result[0] += 0.016805365352725435;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.532256603240968573) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12789058685302912) ) ) {
                          result[0] += 0.002755242055744357;
                        } else {
                          result[0] += -0.019717398122160324;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.216319084167481357) ) ) {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)6.000000000000000888) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.029068946838379794) ) ) {
                              result[0] += 0.022841340495758432;
                            } else {
                              result[0] += -0.06497374637183292;
                            }
                          } else {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += 0.013261381407613721;
                            } else {
                              result[0] += 0.09907791640135932;
                            }
                          }
                        } else {
                          result[0] += 0.06607389024224344;
                        }
                      }
                    } else {
                      result[0] += -0.05259386236716995;
                    }
                  }
                } else {
                  result[0] += -0.0284320313695994;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                result[0] += 0.041342180342845036;
              } else {
                result[0] += -0.0065258117411955245;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.071567356586456743) ) ) {
                result[0] += -0.13635219726535;
              } else {
                result[0] += -0.030795385137134507;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.321723937988282138) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.142630577087403232) ) ) {
              result[0] += 0.04504932716658522;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  result[0] += -0.024914106752458826;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.810334205627442294) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.016581237882179717;
                    } else {
                      result[0] += 0.06110934124137775;
                    }
                  } else {
                    result[0] += -0.01237595148823773;
                  }
                }
              } else {
                result[0] += -0.04164304706243667;
              }
            }
          } else {
            result[0] += -0.10626534945294189;
          }
        }
      } else {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.009367786240991228;
            } else {
              result[0] += -0.0012882266113601987;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.07496595382690607) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.189540147781372958) ) ) {
                result[0] += -0.03422144075070375;
              } else {
                result[0] += -0.004424072580602967;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.433569431304932529) ) ) {
                result[0] += -0.0021102251101329773;
              } else {
                result[0] += -0.07380982730573712;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                result[0] += 0.010039653302792113;
              } else {
                result[0] += -0.027936545747358058;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.088880300521851474) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.025292431714785575;
                } else {
                  result[0] += -0.01320159847365028;
                }
              } else {
                result[0] += 0.04063726663740875;
              }
            }
          } else {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0018191112388607448;
            } else {
              result[0] += -0.03200899591110181;
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
    result[0] += 0.00219370032679958;
  } else {
    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.06868677581279818;
        } else {
          if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.099551913852898;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.736966371536255771) ) ) {
                  result[0] += -0.05964661488664217;
                } else {
                  result[0] += 0.035524836996624364;
                }
              }
            } else {
              result[0] += 0.024464865155029224;
            }
          } else {
            result[0] += 0.058317051014360334;
          }
        }
      } else {
        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.597137451171875888) ) ) {
            if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.059627806674858334;
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.004418962658683961;
                  } else {
                    result[0] += -0.02719051848964117;
                  }
                }
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.64367699623107999) ) ) {
                    result[0] += 0.01681158472454776;
                  } else {
                    result[0] += -0.14218894173552035;
                  }
                } else {
                  result[0] += 0.08198323958351478;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.00396752357482999) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.342454433441162998) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.008986597638215991;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.53326439857482999) ) ) {
                          result[0] += 0.027627434358266496;
                        } else {
                          result[0] += -0.0493590207013042;
                        }
                      }
                    } else {
                      result[0] += -0.09331044282951062;
                    }
                  } else {
                    result[0] += 0.01751408486689155;
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.051608069833713556;
                  } else {
                    if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                        result[0] += 0.008805211669644685;
                      } else {
                        result[0] += -0.02468194897265026;
                      }
                    } else {
                      result[0] += -0.03641439550134082;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                  result[0] += -0.021825497571308767;
                } else {
                  result[0] += 0.0017384354571598941;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.0073215954636712145;
                } else {
                  result[0] += -0.04650312424719626;
                }
              } else {
                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.125962495803833896) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.07822253261597262;
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.381086945533752885) ) ) {
                              result[0] += -0.0613537678627784;
                            } else {
                              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.743430852890015537) ) ) {
                                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += 0.009096726167243404;
                                } else {
                                  result[0] += -0.007329223609123249;
                                }
                              } else {
                                result[0] += 0.045995589060610166;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.851041555404663974) ) ) {
                              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                                result[0] += 0.00037006455432132124;
                              } else {
                                result[0] += 0.10371627630653679;
                              }
                            } else {
                              result[0] += -0.020397224602895322;
                            }
                          } else {
                            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                                if ( LIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
                                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.835998296737671787) ) ) {
                                    result[0] += -0.016437617956565755;
                                  } else {
                                    result[0] += -0.0428263355551325;
                                  }
                                } else {
                                  result[0] += -0.007888607675411675;
                                }
                              } else {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.368446350097658026) ) ) {
                                  result[0] += -0.021427246181300978;
                                } else {
                                  result[0] += 0.036554058905901236;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                                result[0] += 0.12169599026193663;
                              } else {
                                result[0] += 0.009253633452963896;
                              }
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.637949228286744052) ) ) {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += -0.1058607180058548;
                            } else {
                              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += -0.004905091115853709;
                                } else {
                                  result[0] += -0.06708876319586268;
                                }
                              } else {
                                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    result[0] += -0.04609739925182875;
                                  } else {
                                    result[0] += -0.12161580610804851;
                                  }
                                } else {
                                  result[0] += -0.02569568408767523;
                                }
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.04837613179120286;
                            } else {
                              result[0] += -0.1133407538646948;
                            }
                          }
                        } else {
                          result[0] += -0.0014125937734782228;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.08032307509919628;
                      } else {
                        result[0] += -0.034948143303675634;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.602003335952759233) ) ) {
                      result[0] += -0.05959006151657162;
                    } else {
                      result[0] += -0.0004931951973068939;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
                    result[0] += -0.021277368923669007;
                  } else {
                    result[0] += 0.02803615991855889;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.605039834976196733) ) ) {
                result[0] += 0.003591784828953729;
              } else {
                if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.06281967606609369;
                } else {
                  result[0] += -0.021940735410552543;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.02936991440579777;
          } else {
            result[0] += 0.013747026364199909;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
        result[0] += 0.00040165286310561536;
      } else {
        result[0] += -0.09024439028006176;
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += 0.0029791017349428267;
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.158952236175537998) ) ) {
                  result[0] += 0.02329472345676134;
                } else {
                  result[0] += 0.05271169668835956;
                }
              } else {
                result[0] += 0.010727507884942866;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.634540319442749912) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                    result[0] += 0.005898594587703888;
                  } else {
                    result[0] += -0.053162725207096945;
                  }
                } else {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.836270570755005771) ) ) {
                      result[0] += -0.006405354784305936;
                    } else {
                      result[0] += -0.07041456364936574;
                    }
                  } else {
                    result[0] += 0.01206963527402147;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.531673669815064365) ) ) {
                  if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.018493553266052988;
                  } else {
                    result[0] += -0.13903324600355213;
                  }
                } else {
                  result[0] += 0.048631476107376226;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.774904012680054599) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.226807117462159091) ) ) {
              result[0] += -0.0052127566823757104;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.600145101547242099) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
                  result[0] += -0.06157379407276761;
                } else {
                  result[0] += -0.3006578757442945;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.016618835532442294;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.03615116272837953;
                    } else {
                      result[0] += -0.15119042382108092;
                    }
                  } else {
                    result[0] += -0.17523490050730164;
                  }
                }
              }
            }
          } else {
            result[0] += -0.1394556626867085;
          }
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.373361587524414951) ) ) {
            result[0] += -0.009609249094933608;
          } else {
            result[0] += -0.044609329539913006;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.310776710510254794) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                  result[0] += -0.055352936279586165;
                } else {
                  result[0] += 0.050456792318273685;
                }
              } else {
                result[0] += -0.08241693792004468;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.0023450520521764665;
              } else {
                result[0] += 0.0012615196713210777;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                result[0] += 0.02074898514323624;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.017299653549011796;
                } else {
                  result[0] += 0.0033806863990027028;
                }
              }
            } else {
              result[0] += 0.007479617854568909;
            }
          }
        }
      }
    } else {
      result[0] += -0.007125183403067662;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.553712725639343706) ) ) {
      result[0] += -0.007661553565155233;
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.888826131820679155) ) ) {
          result[0] += 0.008694408927785694;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += 0.0007291873353783999;
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0005895120075356418;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.88435244560241788) ) ) {
                        result[0] += 0.006983292330514879;
                      } else {
                        result[0] += 0.08081188048592164;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.637949228286744052) ) ) {
                        result[0] += 0.0015552131354538102;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.780892848968506748) ) ) {
                          result[0] += -0.0283027197982979;
                        } else {
                          result[0] += -0.0022582188055605065;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.241523027420044833) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.018593134864926013;
                    } else {
                      result[0] += 0.012172622538984539;
                    }
                  } else {
                    result[0] += -0.02093074563715469;
                  }
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.921060562133789951) ) ) {
                      result[0] += -0.0053329837827449916;
                    } else {
                      result[0] += -0.04168054859392413;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.182065486907959873) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.241300821304322177) ) ) {
                        result[0] += -0.0019034564339282208;
                      } else {
                        result[0] += 0.02238156962189313;
                      }
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.03580295911395898;
                          } else {
                            result[0] += 0.012040513297771116;
                          }
                        } else {
                          if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.08841433677782756;
                          } else {
                            result[0] += 0.005568900643884492;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                          result[0] += 0.014450962871904272;
                        } else {
                          result[0] += 0.07214558525822962;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.019891364508715925;
                }
              }
            } else {
              if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.53326439857482999) ) ) {
                  result[0] += -0.02090653878480861;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                    result[0] += 0.004225415700754335;
                  } else {
                    if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += -0.0076077978634040274;
                    } else {
                      result[0] += -0.04696120762287967;
                    }
                  }
                }
              } else {
                result[0] += -0.03466991170792255;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += 0.005690205088878887;
          } else {
            result[0] += -0.014884609805383232;
          }
        } else {
          result[0] += -0.035124324284544255;
        }
      }
    }
  }
  if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)1.500000000000000222) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.000403825395420162;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.005805244170984307;
            } else {
              result[0] += -0.03366528259802592;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.525167226791382724) ) ) {
              result[0] += -0.003608378391313391;
            } else {
              result[0] += 0.16701823007924652;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.007174583665848955;
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              result[0] += -0.03238140987103797;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.777674019336700661) ) ) {
                result[0] += 0.11424098056032149;
              } else {
                result[0] += -0.00659528504175415;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
              if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.0035149521743617768;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.778982400894165927) ) ) {
                  result[0] += 0.043108629409238625;
                } else {
                  result[0] += 0.010060999790580763;
                }
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                    result[0] += 0.0016159176063935184;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.749434947967529741) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.777674019336700661) ) ) {
                        result[0] += -0.17810870762963804;
                      } else {
                        result[0] += -0.056034434742195695;
                      }
                    } else {
                      result[0] += -0.020554565535602046;
                    }
                  }
                } else {
                  result[0] += -0.0945757550160376;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
                        result[0] += 0.05382119582881394;
                      } else {
                        result[0] += -0.06912748736257111;
                      }
                    } else {
                      result[0] += 0.015089479859990568;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.329314231872559482) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += -0.1175059405330185;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.357691764831543413) ) ) {
                            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.007245540266995411;
                            } else {
                              result[0] += -0.04989595989405258;
                            }
                          } else {
                            result[0] += 0.07156868233989189;
                          }
                        }
                      } else {
                        result[0] += -0.0931951430826288;
                      }
                    } else {
                      result[0] += -0.0021531259270081136;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.027776920187285233;
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.329747915267945224) ) ) {
                      result[0] += 0.08640589890967709;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.81628036499023615) ) ) {
                        result[0] += 0.024862530299896202;
                      } else {
                        result[0] += -0.11744067123721014;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.737386107444763628) ) ) {
                if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.010379825429904352;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += 0.0714487198781105;
                    } else {
                      result[0] += 0.013371330012323586;
                    }
                  }
                } else {
                  result[0] += -0.049123155586348055;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.623839378356934482) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0063159089099433215;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.718933820724488193) ) ) {
                        result[0] += -0.02237681880595494;
                      } else {
                        result[0] += -0.09812464741637654;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.605039834976196733) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0012572155910928508;
                      } else {
                        result[0] += -0.034801863076535974;
                      }
                    } else {
                      if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.014471930540631221;
                      } else {
                        result[0] += 0.004826269768077963;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.006880818419716528;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.028733754908572336;
                      } else {
                        result[0] += 0.021785975647534386;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.038980097507623754;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.016886373924322542;
                      } else {
                        result[0] += 0.002136934006555587;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.322819471359253818) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.325443029403687412) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.357691764831543413) ) ) {
                      result[0] += -0.09921816080252845;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.131699204444885698) ) ) {
                        result[0] += -0.1253144975044199;
                      } else {
                        result[0] += -0.011604839299741397;
                      }
                    }
                  } else {
                    result[0] += 0.02708650814270735;
                  }
                } else {
                  result[0] += 0.03534652384566164;
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.03478412848806907;
                    } else {
                      result[0] += 0.08584122439952904;
                    }
                  } else {
                    result[0] += -0.032777996656368144;
                  }
                } else {
                  result[0] += 0.03508291546829045;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.006828753904962213;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.58713245391845881) ) ) {
      result[0] += -0.0001804994131581413;
    } else {
      if ( UNLIKELY( !(data[28].missing != -1) || (data[28].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          result[0] += -0.009679845320320384;
        } else {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.004341244529767984;
            } else {
              result[0] += 0.017492614355420925;
            }
          } else {
            result[0] += -0.04011196903864748;
          }
        }
      } else {
        result[0] += -0.01418176844872747;
      }
    }
  }
}

