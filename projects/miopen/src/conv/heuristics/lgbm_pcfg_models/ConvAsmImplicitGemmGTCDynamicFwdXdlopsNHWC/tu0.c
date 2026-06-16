
#include "header.h"

void predict_unit0(union Entry* data, double* result) {
  unsigned int tmp;
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.03322894407503301;
                    } else {
                      result[0] += -0.11164845577827553;
                    }
                  } else {
                    result[0] += 0.09755713743436761;
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.09787366032979246;
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                        result[0] += -0.14675695719755674;
                      } else {
                        result[0] += 0.06861866563246004;
                      }
                    }
                  } else {
                    result[0] += 0.13466096837174252;
                  }
                }
              } else {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.11462564692312688;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
                    result[0] += 0.042649109515193705;
                  } else {
                    result[0] += -0.11822953456697763;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
                    result[0] += 0.05070795989750765;
                  } else {
                    result[0] += 0.12146651252892365;
                  }
                } else {
                  result[0] += -0.028469460652708556;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.17679772805864383;
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                        result[0] += -0.14779718260389338;
                      } else {
                        result[0] += 0.11143157720794994;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += -0.11807107774657291;
                      } else {
                        result[0] += 0.06325143143520826;
                      }
                    }
                  } else {
                    result[0] += 0.16586276521053728;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
              result[0] += 0.13017609114246317;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
                  result[0] += -0.13427836413925803;
                } else {
                  result[0] += 0.04856833631014934;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)159.5000000000000284) ) ) {
                  result[0] += 0.16722724024918575;
                } else {
                  result[0] += -0.1208107552385541;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.04048063260551896;
                } else {
                  result[0] += 0.10108203179023817;
                }
              } else {
                result[0] += -0.0201698887468447;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.05166806696589119;
              } else {
                result[0] += 0.0591810411459126;
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)201.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.0122437013996481;
                } else {
                  result[0] += -0.15469050294233178;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.024270281173905183;
                  } else {
                    result[0] += 0.06529380954131674;
                  }
                } else {
                  result[0] += -0.08280455073151677;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.11957718895753243;
                } else {
                  result[0] += -0.09474528846008856;
                }
              } else {
                result[0] += -0.17754040763121906;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338887453079224521) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += 0.11895939185431235;
              } else {
                result[0] += -0.05816485360407998;
              }
            } else {
              result[0] += 0.17116340395965846;
            }
          } else {
            result[0] += -0.11232982457982994;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            result[0] += -0.1600481439566207;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.12964510814111943;
            } else {
              result[0] += 0.07316908083751546;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
          result[0] += -0.06281485263884798;
        } else {
          result[0] += -0.16104737594581642;
        }
      } else {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
            result[0] += 0.08639596990189267;
          } else {
            result[0] += -0.08490367928151082;
          }
        } else {
          result[0] += 0.1127383166661588;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.012810926859352421;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
              result[0] += -0.10737232131987062;
            } else {
              result[0] += 0.0020817937637453575;
            }
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.610357046127320224) ) ) {
            if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.1686709016086124;
            } else {
              result[0] += -0.10840592760373483;
            }
          } else {
            result[0] += -0.0572580032687406;
          }
        }
      } else {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.551017761230469638) ) ) {
            result[0] += -0.005646031523320338;
          } else {
            result[0] += -0.10164002653349236;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.040189959971633514;
            } else {
              result[0] += 0.09419503061667892;
            }
          } else {
            result[0] += 0.009515342820626266;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.00385427814851633;
        } else {
          result[0] += -0.09501567691880475;
        }
      } else {
        result[0] += -0.17476829170743663;
      }
    }
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.06981228949886276;
            } else {
              result[0] += -0.019851044825374024;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.824383735656740058) ) ) {
              if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.041339577637753366;
              } else {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.12738325941454293;
                  } else {
                    result[0] += -0.0519835622055987;
                  }
                } else {
                  result[0] += 0.020659595194070065;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  result[0] += -0.07224226974460533;
                } else {
                  result[0] += -0.14258085241365123;
                }
              } else {
                result[0] += -0.1526620682699581;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)164.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.11596583315586309;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.014964350839184926;
                } else {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.06296248958819725;
                  } else {
                    result[0] += -0.14136878564535255;
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.1477606032862331;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)49.50000000000000711) ) ) {
                          result[0] += -0.023438284296595308;
                        } else {
                          result[0] += 0.059642904432191494;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)130.5000000000000284) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                              result[0] += -0.13645436242676318;
                            } else {
                              result[0] += -0.038049734737701985;
                            }
                          } else {
                            result[0] += 0.005766863573445097;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
                            result[0] += 0.031541817867452616;
                          } else {
                            result[0] += -0.07569625951516316;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                          result[0] += 0.023866794448305462;
                        } else {
                          result[0] += -0.10592623998686466;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                            result[0] += 0.027140944954990455;
                          } else {
                            result[0] += 0.08038689848153963;
                          }
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                            result[0] += -0.08751037053116241;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                              result[0] += -0.037026098109326104;
                            } else {
                              result[0] += 0.09678655042391807;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.16785589768224293;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.1265535984630064;
                  } else {
                    result[0] += -0.030283900687268328;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
              result[0] += -0.005342539431260052;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.08243476804063636;
                  } else {
                    result[0] += -0.030186917955227124;
                  }
                } else {
                  result[0] += -0.09111463906608804;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06967335253545402;
                } else {
                  result[0] += -0.14937020140463977;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.1562755360011078;
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)27.50000000000000355) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.11161539989570401;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.16594791412353693) ) ) {
              result[0] += 0.07474494504013493;
            } else {
              result[0] += -0.08373022989934942;
            }
          }
        } else {
          result[0] += 0.12976503072166062;
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
            result[0] += 0.07394229390672895;
          } else {
            result[0] += 0.00669460330062734;
          }
        } else {
          result[0] += -0.09870930443736078;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.04203094654159126;
              } else {
                result[0] += -0.05177369568585205;
              }
            } else {
              result[0] += -0.08087193040522941;
            }
          } else {
            result[0] += 0.09297914479988423;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.11857799759006747;
            } else {
              result[0] += -0.10120094494667148;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.06352361590190317;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.10511904595903723;
              } else {
                result[0] += 0.16031217980977208;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
              result[0] += 0.057347046322744094;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.020485025959220925;
              } else {
                result[0] += -0.12335155384465005;
              }
            }
          } else {
            result[0] += 0.08907129585941662;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.018568789005228414;
            } else {
              result[0] += 0.0755199248295015;
            }
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.15228934124195417;
            } else {
              result[0] += -0.06496965425959934;
            }
          }
        }
      }
    } else {
      result[0] += -0.14739812808687;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
        result[0] += 0.02563672671337355;
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += 0.00699281630581056;
            } else {
              result[0] += -0.075479167046549;
            }
          } else {
            result[0] += -0.10829497933390674;
          }
        } else {
          result[0] += -0.13758636043142466;
        }
      }
    } else {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.12748661220804627;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)29.50000000000000355) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)17.50000000000000355) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                      result[0] += -0.08105847767381158;
                    } else {
                      result[0] += -0.0016359073425547144;
                    }
                  } else {
                    result[0] += -0.07682369478654688;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.06907978023707073;
                  } else {
                    if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.06944594547346833;
                    } else {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.027219982606430007;
                      } else {
                        result[0] += 0.021531493689983083;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                  result[0] += -0.1313985677207011;
                } else {
                  result[0] += -0.02131645329472757;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  result[0] += -0.009857414287772759;
                } else {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.12394910838587943;
                  } else {
                    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                      result[0] += 0.0058601708659448744;
                    } else {
                      result[0] += -0.06111085064093148;
                    }
                  }
                }
              } else {
                result[0] += -0.11201424058534835;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.1452246359789783;
          } else {
            result[0] += -0.04556814786578085;
          }
        }
      } else {
        result[0] += -0.1183622894918619;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.463993549346925604) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)232.5000000000000284) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
                  if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.021394597330945703;
                  } else {
                    result[0] += -0.054713323721335465;
                  }
                } else {
                  result[0] += 0.045034981304749644;
                }
              } else {
                result[0] += 0.0938155982824075;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += 0.00020683881318991744;
              } else {
                result[0] += -0.10552369587792304;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.02997638026435794;
            } else {
              result[0] += 0.10896192093809334;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.05604881671510597;
              } else {
                result[0] += -0.07564962884469945;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.03177054498470372;
              } else {
                result[0] += 0.09698535710592357;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.126931190490723544) ) ) {
                result[0] += 0.0327308000208784;
              } else {
                result[0] += 0.10768305716115308;
              }
            } else {
              result[0] += 0.1444750376823106;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                  result[0] += -0.11558266433514637;
                } else {
                  result[0] += -0.01664859317885322;
                }
              } else {
                result[0] += -0.12639620588774453;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.018142334058389786;
              } else {
                result[0] += -0.10349493638309717;
              }
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.006028509757769567;
            } else {
              result[0] += -0.1135929755568264;
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
              result[0] += 0.10852927429640397;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.04353813593283784;
                } else {
                  result[0] += 0.04916488449038192;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.020127415657043901) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.883974552154541904) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)146.5000000000000284) ) ) {
                          if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.04206120250543885;
                          } else {
                            result[0] += -0.02795488362821226;
                          }
                        } else {
                          result[0] += 0.08054359683074806;
                        }
                      } else {
                        result[0] += 0.07297333094365967;
                      }
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                        result[0] += 0.11012306941537751;
                      } else {
                        result[0] += 0.05014248760926056;
                      }
                    }
                  } else {
                    result[0] += -0.08739541215615212;
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
                        result[0] += 0.0088824872317519;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += 0.1080884708191528;
                        } else {
                          result[0] += -0.050067773871647375;
                        }
                      }
                    } else {
                      result[0] += 0.10373217074226333;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.07947867712828532;
                    } else {
                      result[0] += 0.04886236483843316;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.06435433862187699;
          }
        }
      }
    } else {
      result[0] += -0.14862221866038058;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += 0.010257378804691495;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.79288959503174006) ) ) {
              result[0] += -0.0496990971466319;
            } else {
              result[0] += 0.07217841657473843;
            }
          } else {
            result[0] += -0.10720364433781222;
          }
        }
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.043221083703912824;
            } else {
              result[0] += -0.11404845913281812;
            }
          } else {
            result[0] += -0.13017447195992873;
          }
        } else {
          result[0] += -0.06489721048190887;
        }
      }
    } else {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.11138214336272337;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)29.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.004059041248897528;
                  } else {
                    result[0] += -0.10755087521091543;
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.342858314514161933) ) ) {
                    result[0] += -0.020514721261193813;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.05007074250259522;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += -0.1555637464721583;
                      } else {
                        result[0] += 3.318913221029006e-05;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.0196822431274644;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)212.5000000000000284) ) ) {
                if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.07689956916565116;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
                      result[0] += -0.0702176746321788;
                    } else {
                      result[0] += 0.011942827295235041;
                    }
                  }
                } else {
                  result[0] += -0.10310404398345825;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.01897144452215257;
                  } else {
                    result[0] += -0.1360835162635826;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.09500838242175481;
                  } else {
                    result[0] += 0.05596447719272758;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.10259268138023275;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.014757649373830922;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)70.50000000000001421) ) ) {
                      result[0] += -0.0095215425299114;
                    } else {
                      result[0] += -0.11031477059000615;
                    }
                  }
                }
              } else {
                result[0] += -0.09922578816628384;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.14610535777337919;
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.134879350662232333) ) ) {
              result[0] += -0.09780681354343662;
            } else {
              result[0] += 0.08760487702181757;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)47.50000000000000711) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
            result[0] += -0.13487389344183706;
          } else {
            result[0] += -0.02395261803871586;
          }
        } else {
          result[0] += -0.1222636899772691;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.972562313079834873) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.07730369754547234;
                  } else {
                    result[0] += 0.009135779069561494;
                  }
                } else {
                  result[0] += 0.008004061748583443;
                }
              } else {
                result[0] += 0.09475354339868754;
              }
            } else {
              result[0] += 0.031134002151268565;
            }
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
              result[0] += 0.0022544037950964204;
            } else {
              result[0] += 0.0502728749757069;
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.04869754963562111;
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.06056619862375602;
            } else {
              result[0] += 0.10908051614164574;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.687107801437378818) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
                  result[0] += -0.09805972272047447;
                } else {
                  result[0] += -0.006658307748385491;
                }
              } else {
                result[0] += -0.11969071311709994;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.011074523247626797;
              } else {
                result[0] += -0.117760208904629;
              }
            }
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.0005591084005650917;
            } else {
              result[0] += -0.11766501892271382;
            }
          }
        } else {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)27.50000000000000355) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.872101783752442294) ) ) {
                    result[0] += -0.004141643018833268;
                  } else {
                    result[0] += 0.0946457446169801;
                  }
                } else {
                  result[0] += -0.09174579837493689;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
                    result[0] += 0.020877822838380883;
                  } else {
                    result[0] += 0.11212243465263907;
                  }
                } else {
                  result[0] += 0.10939025653841135;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.652390718460083896) ) ) {
                result[0] += 0.023629000584181964;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.054897287578702575;
                } else {
                  result[0] += -0.06334115615425409;
                }
              }
            }
          } else {
            result[0] += -0.06917620633145627;
          }
        }
      }
    } else {
      result[0] += -0.13469534681491382;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.06419420761546392;
          } else {
            result[0] += 0.02930305838285112;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += -0.03078146392663593;
          } else {
            result[0] += -0.09961925625550855;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)135.5000000000000284) ) ) {
            result[0] += -0.08954974769347042;
          } else {
            result[0] += -0.018144939096337475;
          }
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.02845032871822872;
            } else {
              result[0] += -0.1109075132984647;
            }
          } else {
            result[0] += -0.1332401333021932;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.09572917719540003;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.06923408337983682;
                } else {
                  result[0] += 0.023513973546600404;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.15419319691954186;
                } else {
                  result[0] += 0.00987846953045619;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.11065035481192108;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
                  result[0] += -0.09167161533317486;
                } else {
                  result[0] += 0.03070855472486078;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.0391194039334345;
              } else {
                result[0] += -0.11296309064587287;
              }
            } else {
              result[0] += 0.08152930164635017;
            }
          } else {
            result[0] += -0.13710437900192726;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
            if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.05618204903684668;
            } else {
              result[0] += -0.028440379228430897;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.052798362959658146;
            } else {
              result[0] += -0.11828256441164986;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.612332344055176669) ) ) {
            result[0] += -0.04570914022135935;
          } else {
            result[0] += -0.13215039119474617;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.80459928512573331) ) ) {
                result[0] += -0.09850040840401832;
              } else {
                result[0] += 0.042738744186018064;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
                result[0] += 0.005819351468875528;
              } else {
                result[0] += 0.08604500974005867;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.0001441658082079105;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.138696432113648349) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.08815744149431126;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                      result[0] += -0.04781552471695544;
                    } else {
                      result[0] += 0.06682758540600343;
                    }
                  }
                } else {
                  result[0] += -0.04301930226611724;
                }
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.09163930053754926;
                  } else {
                    result[0] += 0.023048564123354234;
                  }
                } else {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06298674082206081;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.10276228143917494;
                      } else {
                        result[0] += 0.004998228972390249;
                      }
                    } else {
                      result[0] += 0.12483098820414736;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.824383735656740058) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                result[0] += -0.06335037183159706;
              } else {
                result[0] += 0.03911227095006119;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.11144738448719525;
                  } else {
                    result[0] += 0.015801803300014763;
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.12584565865318062;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      result[0] += -0.054542853062633706;
                    } else {
                      result[0] += 0.04049046645429128;
                    }
                  }
                }
              } else {
                result[0] += -0.1347376999179192;
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)27.50000000000000355) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                    result[0] += 0.002555072783622631;
                  } else {
                    result[0] += 0.08313638043974794;
                  }
                } else {
                  result[0] += -0.11536747037625912;
                }
              } else {
                result[0] += 0.0875246344963676;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.01194855002138081;
                } else {
                  result[0] += 0.030710444363090297;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.046300590233137165;
                } else {
                  result[0] += -0.10810499095959712;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
            result[0] += 0.08934055765517125;
          } else {
            result[0] += -0.0024837314206740944;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            result[0] += -0.01750922747484552;
          } else {
            result[0] += -0.12255788164117064;
          }
        }
      }
    } else {
      result[0] += -0.12499864382671888;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.10533721204670753;
              } else {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.028778894005842916;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                          result[0] += -0.010450783376765863;
                        } else {
                          result[0] += -0.11016102703179718;
                        }
                      } else {
                        result[0] += 0.0030598306519910475;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.014831542968751776) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                          result[0] += 0.07821298357214751;
                        } else {
                          result[0] += -0.041264024480457226;
                        }
                      } else {
                        result[0] += -0.08046977787969677;
                      }
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.12229920149700446;
                      } else {
                        result[0] += -0.03865547990099033;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                    result[0] += 0.013763686395051215;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.0014434260377727626;
                    } else {
                      result[0] += -0.08301623500663419;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.05720260279753814;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.238486170768738237) ) ) {
                    result[0] += 0.05702926344349477;
                  } else {
                    result[0] += -0.037355311222805256;
                  }
                } else {
                  result[0] += -0.07764900452036727;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
              result[0] += -0.024360619883813443;
            } else {
              result[0] += -0.08291933926412622;
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.12903261612661934;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.099201440811158115) ) ) {
              result[0] += 0.025970999094740124;
            } else {
              result[0] += -0.13385129429769013;
            }
          }
        }
      } else {
        result[0] += -0.09595770673249827;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += 0.007530822069976253;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.10714166580237523;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0009330774670946046;
            } else {
              result[0] += -0.10412100530800344;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.13461822124262027;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03832131280979909;
            } else {
              result[0] += -0.10904028151438286;
            }
          } else {
            result[0] += -0.11063981645046675;
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.870205879211427558) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)232.5000000000000284) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.223295450210572177) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                result[0] += -0.07410711207337181;
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.03936065079876595;
                } else {
                  result[0] += -0.014658149415013222;
                }
              }
            } else {
              result[0] += 0.03349552760468646;
            }
          } else {
            result[0] += 0.06732627007944436;
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.008752019824243213;
          } else {
            result[0] += 0.08082388196462871;
          }
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
          result[0] += 0.10160542020034394;
        } else {
          if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.048097633195438916;
            } else {
              result[0] += -0.06096290685306484;
            }
          } else {
            result[0] += 0.07398478796337382;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)75.50000000000001421) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.652390718460083896) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03747279339408873;
              } else {
                result[0] += -0.05358858499847401;
              }
            } else {
              result[0] += 0.07371164277543497;
            }
          } else {
            result[0] += -0.09842088429330421;
          }
        } else {
          result[0] += 0.07417920585259125;
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.267176389694214755) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
                result[0] += -0.03077572229989492;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.05120020695449984;
                  } else {
                    result[0] += -0.07601524250122166;
                  }
                } else {
                  result[0] += -0.016587491851642776;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.09392488954488104;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.005815278670816914;
                } else {
                  result[0] += -0.12311334785898875;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)149.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += -0.031656264283193014;
                } else {
                  result[0] += 0.0364400690808838;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)180.5000000000000284) ) ) {
                  result[0] += -0.1269289290961099;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                      result[0] += -0.08854285337915109;
                    } else {
                      result[0] += 0.017921504366362206;
                    }
                  } else {
                    result[0] += -0.0005849206281746163;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.07870842733136481;
              } else {
                result[0] += -0.10337781648686933;
              }
            }
          }
        } else {
          result[0] += -0.09484920024470579;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.07001414075819895;
            } else {
              result[0] += 0.012347162728614;
            }
          } else {
            result[0] += -0.057004793119082145;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.05126802386456475;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += 0.06337995405348734;
            } else {
              result[0] += -0.03950404332613706;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.005100365991117801;
            } else {
              result[0] += -0.10465794998496376;
            }
          } else {
            result[0] += -0.09070417285163446;
          }
        } else {
          result[0] += -0.09922474830550893;
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.09908650856082127;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02967933086669244;
                    } else {
                      result[0] += 0.052842457820219416;
                    }
                  } else {
                    result[0] += -0.07865302128601429;
                  }
                } else {
                  result[0] += 0.010122103477724286;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.099201440811158115) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                      result[0] += 0.062735295321201;
                    } else {
                      result[0] += -0.13493480298103327;
                    }
                  } else {
                    result[0] += -0.08952155912938797;
                  }
                } else {
                  result[0] += -0.12153123227494833;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                result[0] += -0.1058718194657049;
              } else {
                result[0] += -0.027324341778137654;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.12299316146999212;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
              result[0] += -0.05779857287845891;
            } else {
              result[0] += 0.08005604704878663;
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.04857568306717936;
            } else {
              result[0] += -0.02442376671582016;
            }
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.0978747422852836;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.024196677058351452;
              } else {
                result[0] += -0.0859785314157653;
              }
            }
          }
        } else {
          result[0] += -0.10440539025666801;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.448499202728272373) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += 0.011994257929982342;
                } else {
                  result[0] += -0.06131954481795146;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.280659198760987216) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.04850727051152942;
                  } else {
                    result[0] += -0.030099539236289732;
                  }
                } else {
                  result[0] += 0.04969431603019278;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.036412608914850896;
              } else {
                result[0] += 0.07772825431198493;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.009961629785022045;
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.04260090538764999;
              } else {
                result[0] += 0.08572395760576268;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  result[0] += -0.04541644998871486;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.055668660105398796;
                      } else {
                        result[0] += 0.00789186673588815;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                        result[0] += 0.01960768340681909;
                      } else {
                        result[0] += -0.10293617969551264;
                      }
                    }
                  } else {
                    result[0] += 0.060277821178596175;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.980170249938965732) ) ) {
                  result[0] += 0.040429605186717184;
                } else {
                  result[0] += -0.09618915845382739;
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                result[0] += 0.06580412834459949;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                  result[0] += 0.045594350751132884;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.12948420570565589;
                  } else {
                    result[0] += 0.004242669512457005;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04943484387696069;
                } else {
                  result[0] += -0.11310793149949248;
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.022127523305155564;
                } else {
                  result[0] += -0.10140779506261917;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += -0.09862829892116398;
              } else {
                result[0] += 0.0028038810674360926;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.611996650695801669) ) ) {
              result[0] += 0.019116534263439157;
            } else {
              result[0] += 0.12960111360921045;
            }
          } else {
            result[0] += -0.01871950648615609;
          }
        } else {
          result[0] += -0.0803462393753294;
        }
      }
    } else {
      result[0] += -0.11330213914455389;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.006137558135199929;
          } else {
            result[0] += -0.08592221315537968;
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.11708102259278615;
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.099201440811158115) ) ) {
              result[0] += 0.028822729078813832;
            } else {
              result[0] += -0.12473221217552355;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
              result[0] += 0.015002671009840807;
            } else {
              result[0] += -0.08670187979970832;
            }
          } else {
            result[0] += -0.11321129859367285;
          }
        } else {
          result[0] += -0.09115276642609182;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
        if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0430057947037567;
          } else {
            result[0] += -0.014319222769453719;
          }
        } else {
          result[0] += -0.07108490904907484;
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.10284283223333689;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.016253567098990365;
            } else {
              result[0] += -0.0532851041202023;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                result[0] += -0.058573222471992585;
              } else {
                result[0] += -0.10548995180583252;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.015658208090406776;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.008588470451963261;
                  } else {
                    result[0] += -0.10401894131937557;
                  }
                } else {
                  result[0] += -0.10025794278396204;
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.448499202728272373) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)26.50000000000000355) ) ) {
                      result[0] += -0.07649511502469969;
                    } else {
                      result[0] += 0.0336776430145967;
                    }
                  } else {
                    result[0] += -0.04422233649945548;
                  }
                } else {
                  result[0] += 0.02590692684261698;
                }
              } else {
                result[0] += -0.06688324901398954;
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += 0.05298053471305572;
              } else {
                result[0] += -0.02495159851075113;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.026032541836161178;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
                result[0] += 0.019055293121847738;
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.07426548187157744;
                } else {
                  result[0] += -0.049109293281390826;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
            result[0] += 0.08862024662847695;
          } else {
            if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.0322153187416905;
              } else {
                result[0] += -0.06540343114673608;
              }
            } else {
              result[0] += 0.06353969934948804;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
            result[0] += -0.06214911140553784;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.109245061874390537) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.05557900378113959;
                    } else {
                      result[0] += 0.033247182603080655;
                    }
                  } else {
                    result[0] += -0.04846693495412742;
                  }
                } else {
                  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.053180458544933445;
                  } else {
                    result[0] += 0.0030654122398280133;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.126931190490723544) ) ) {
                  result[0] += 0.02997895130225217;
                } else {
                  result[0] += -0.1073475902215269;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.448499202728272373) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      result[0] += 0.016144459632154166;
                    } else {
                      result[0] += -0.06723914275887663;
                    }
                  } else {
                    result[0] += 0.042816458471129644;
                  }
                } else {
                  result[0] += 0.08301965370397;
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.206746339797974521) ) ) {
                  result[0] += 0.07155535869208215;
                } else {
                  if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.026856921004062978;
                  } else {
                    result[0] += 0.0654184496437342;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.06336825232900874;
              } else {
                result[0] += -0.01237207240923;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.005655361740504758;
                } else {
                  result[0] += -0.08360394587703754;
                }
              } else {
                result[0] += 0.022456064092422998;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += -0.09397020986435065;
              } else {
                if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.05459653299537969;
                } else {
                  result[0] += 0.01511753706960235;
                }
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
                  result[0] += 0.01161206947790893;
                } else {
                  result[0] += -0.06969059060518068;
                }
              } else {
                result[0] += -0.10610409906330931;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.10699759168830336;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.004534841255908892;
          } else {
            result[0] += -0.07986526754586275;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.11245301004779627;
          } else {
            result[0] += 0.0017579364596775377;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
              result[0] += 0.016768475014127805;
            } else {
              result[0] += -0.08046779945080135;
            }
          } else {
            result[0] += -0.10534443037177138;
          }
        } else {
          result[0] += -0.09031558072903154;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.09272027093116125;
        } else {
          if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.03135089917269567;
            } else {
              result[0] += -0.013736519888581801;
            }
          } else {
            result[0] += -0.07019403211459947;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.024311445244674382;
          } else {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.02651611427263952;
            } else {
              result[0] += -0.07566631540924261;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.511434078216553178) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                  result[0] += -0.029213610230121468;
                } else {
                  result[0] += -0.08267469900806912;
                }
              } else {
                result[0] += -0.10024726071846567;
              }
            } else {
              result[0] += -0.10483308106752563;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.01717564388260312;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                    result[0] += -0.010876795887907739;
                  } else {
                    result[0] += -0.06341309538600277;
                  }
                } else {
                  result[0] += -0.0998363786250197;
                }
              } else {
                result[0] += -0.08964117617553244;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.337269306182862216) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.02396207951608359;
                } else {
                  result[0] += -0.03976699145322903;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.028231167972487283;
                } else {
                  result[0] += -0.07876575223024988;
                }
              }
            } else {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.03115842636709555;
                } else {
                  result[0] += 0.07436000090438498;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                  result[0] += -0.05045686980041965;
                } else {
                  result[0] += 0.12061198789604731;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.524927973747253862) ) ) {
                result[0] += 0.009143219329775477;
              } else {
                result[0] += 0.08859428717563161;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)204.5000000000000284) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.05290239052787285;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.1084118849600913;
                  } else {
                    result[0] += 0.017991404766486484;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)214.5000000000000284) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                    result[0] += -0.0670577128185187;
                  } else {
                    result[0] += 0.0919298838113213;
                  }
                } else {
                  result[0] += 0.04845338427882816;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.019835097765126704;
                  } else {
                    result[0] += -0.05318446770618486;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.04344606674617001;
                  } else {
                    result[0] += -0.002878428190395235;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += 0.033459740971669455;
                } else {
                  result[0] += -0.11099455885698083;
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
                result[0] += 0.050896842116508;
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += 0.010545151997272548;
                } else {
                  result[0] += -0.11000445865783824;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.10589755318233095;
                  } else {
                    result[0] += -0.02009155143649198;
                  }
                } else {
                  result[0] += -0.10909396420635585;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.022922016635124637;
                } else {
                  result[0] += -0.09447537178367649;
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                result[0] += 0.02569806865005214;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.1111651296161075;
                } else {
                  result[0] += -0.00960810282926153;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
            result[0] += 0.03008645210919127;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.004507402791093128;
            } else {
              result[0] += -0.07893047562165073;
            }
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            result[0] += -0.03204641877940353;
          } else {
            result[0] += -0.10385485473517508;
          }
        }
      }
    } else {
      result[0] += -0.10215633653032097;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.005872581201550471;
              } else {
                result[0] += -0.09796290862309659;
              }
            } else {
              result[0] += -0.05664436611849965;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += -0.0022837126170689235;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.003843954423172714;
                  } else {
                    result[0] += -0.07503202372061932;
                  }
                } else {
                  result[0] += -0.057510363990721396;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.09144170302879974;
                } else {
                  result[0] += -0.01868116726891798;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
              result[0] += 0.06734144282944897;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += 0.10035476683826416;
              } else {
                result[0] += -0.04193926094202427;
              }
            }
          } else {
            result[0] += -0.11725467853082863;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.975242614746095526) ) ) {
          result[0] += -0.02156323113871481;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.04686622060727836;
          } else {
            result[0] += -0.12063219850314706;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          result[0] += 0.010494354808466373;
        } else {
          result[0] += -0.03833667837870843;
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.11993587706015747;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.04755222117042219;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.396947860717774326) ) ) {
              result[0] += -0.0161085796109359;
            } else {
              result[0] += -0.11249936409296735;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                result[0] += -0.0023363655600618614;
              } else {
                result[0] += 0.04853609661654085;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.04595690432140642;
              } else {
                result[0] += -0.07765801244788074;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.017055507175036014;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.09079099654680223;
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.050902802167345045;
                  } else {
                    result[0] += 0.0033737982189329485;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.004461310146849379;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.056896819422439585;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
                      result[0] += 0.05823099246571343;
                    } else {
                      result[0] += 0.10105683627378244;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)184.5000000000000284) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  result[0] += -0.03968033120714465;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.032733210495176766;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        result[0] += 0.01814557945109729;
                      } else {
                        result[0] += -0.12086566643222561;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.854362010955811435) ) ) {
                        result[0] += -0.020864383841423705;
                      } else {
                        result[0] += 0.07166006007103227;
                      }
                    } else {
                      result[0] += 0.06397331787235586;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += 0.07096055624495479;
                } else {
                  result[0] += -0.11338773478469757;
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.08954923270490209;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.0076840234483805505;
                    } else {
                      result[0] += 0.07405800839042818;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.07339919706035578;
                  } else {
                    result[0] += -0.047173050723827756;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                  result[0] += 0.04739886204775087;
                } else {
                  result[0] += -0.028244491333719188;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                result[0] += -0.05679789918357212;
              } else {
                result[0] += 0.03256994842365962;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += -0.07061028043076402;
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.06233512290937875;
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.03914181034082173;
                    } else {
                      result[0] += 0.03213777223180366;
                    }
                  }
                }
              } else {
                result[0] += -0.11908996121427036;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.07395951520714177;
            } else {
              result[0] += 0.02020930191733296;
            }
          } else {
            result[0] += 0.030115745029742166;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            result[0] += 0.0006075784295772813;
          } else {
            result[0] += -0.10316058621976316;
          }
        }
      }
    } else {
      result[0] += -0.0971441118395926;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0023940010421659302;
            } else {
              result[0] += -0.06408596636102472;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                result[0] += 0.055572434370180626;
              } else {
                result[0] += -0.07479405247671926;
              }
            } else {
              result[0] += -0.11528939350429585;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            result[0] += -0.019386942472997475;
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0005607487500090376;
              } else {
                result[0] += -0.10288684916546126;
              }
            } else {
              result[0] += -0.10006956267955201;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.07292526429314658;
            } else {
              result[0] += -0.01299033513593676;
            }
          } else {
            result[0] += -0.0618348822142788;
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.09819555335919072;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.004141899394541544;
              } else {
                result[0] += -0.05076864771939761;
              }
            } else {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  result[0] += -0.02410235631248807;
                } else {
                  result[0] += -0.09048112192496587;
                }
              } else {
                result[0] += -0.04093393850934678;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.10536329893682778;
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)214.5000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                  result[0] += -0.0684289827353344;
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0275599920183594;
                  } else {
                    result[0] += -0.027288727524260144;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                  result[0] += -0.02672319998012805;
                } else {
                  result[0] += 0.07602469251096043;
                }
              }
            } else {
              result[0] += 0.03905707502213443;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.344720840454102451) ) ) {
                result[0] += 0.04381933179227911;
              } else {
                result[0] += -0.04680408188316847;
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += 0.07305554421788783;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.09032034022499562;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.03056064804147478;
                    } else {
                      result[0] += 0.031149598935668626;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.004132022009378089;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.05645231620689709;
                  } else {
                    result[0] += 0.08909666788837335;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)184.5000000000000284) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                    if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.04651239821311568;
                    } else {
                      result[0] += -0.005899874596283597;
                    }
                  } else {
                    result[0] += -0.08825267516884959;
                  }
                } else {
                  result[0] += -0.038403491577319926;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.038502174547002915;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.058067010999294524;
                    } else {
                      result[0] += 0.07015174621595581;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.03138733901394706;
                  } else {
                    result[0] += -0.11410345412661854;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.060491576955366014;
                  } else {
                    result[0] += 0.05651770529917997;
                  }
                } else {
                  result[0] += 0.06485703204726423;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                  result[0] += 0.037751645089041275;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.01027938386379859;
                    } else {
                      result[0] += -0.09330891369571132;
                    }
                  } else {
                    result[0] += -0.1133120868729956;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                result[0] += -0.045905232372709076;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.04159951983015969;
                  } else {
                    result[0] += -0.033122407273887926;
                  }
                } else {
                  result[0] += -0.03154675178443492;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += -0.07714512140439321;
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.07686481336555273;
                  } else {
                    result[0] += -0.004342100033414231;
                  }
                }
              } else {
                result[0] += -0.11794171344836327;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 4.21046325334204e-05;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.048558296432071255;
            } else {
              result[0] += 0.06391251048435605;
            }
          } else {
            result[0] += -0.10106247396746781;
          }
        }
      }
    } else {
      result[0] += -0.09299588005920262;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0053562251954480875;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.08318534267361193;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)17.50000000000000355) ) ) {
                  result[0] += 0.002920382372575333;
                } else {
                  result[0] += -0.05460701321955409;
                }
              }
            }
          } else {
            result[0] += -0.05872908429556398;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += -0.10233028626950513;
          } else {
            result[0] += 0.011907022734767545;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.767332553863526279) ) ) {
              result[0] += 0.01356819522018983;
            } else {
              result[0] += -0.08078413135044078;
            }
          } else {
            result[0] += -0.09898997117835753;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
            result[0] += -0.042707799014720015;
          } else {
            result[0] += -0.09316313260972853;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.055963080805022096;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.0530047737495154;
                } else {
                  if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.032632970794189795;
                  } else {
                    result[0] += -0.005626508127815882;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.09828176027656221;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.007290840148926669) ) ) {
                    result[0] += -0.03771451516686831;
                  } else {
                    result[0] += 0.09540684130324471;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.04119953145553302;
                } else {
                  result[0] += -0.10764815921594789;
                }
              } else {
                result[0] += -0.014342238100152935;
              }
            }
          }
        } else {
          result[0] += -0.058451801803504656;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += 0.07070620902694347;
          } else {
            result[0] += -0.02705176680415506;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.11652253555226291;
              } else {
                result[0] += -0.05534852362633195;
              }
            } else {
              result[0] += -0.09565724681097315;
            }
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.08636458378086434;
            } else {
              result[0] += -0.03978501327722734;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.31402075290679976) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.07962825680254751;
                } else {
                  result[0] += 0.03436985843159076;
                }
              } else {
                result[0] += -0.04423597222298342;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.043579859339082955;
                  } else {
                    result[0] += -0.020708689939889415;
                  }
                } else {
                  result[0] += 0.07407307463429073;
                }
              } else {
                result[0] += -0.012014364635405965;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.344720840454102451) ) ) {
                result[0] += 0.04909345935510334;
              } else {
                result[0] += -0.030883499041995468;
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.07015871418564967;
                } else {
                  result[0] += 0.027787977814499078;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 4.343032384933427e-05;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.04765442905990329;
                  } else {
                    result[0] += 0.07903187671868836;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.0389362603731474;
                } else {
                  result[0] += -0.10071473385908628;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.023893820081152265;
                } else {
                  result[0] += -0.08765900827882026;
                }
              }
            } else {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                result[0] += 0.020432074819363728;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)237.5000000000000284) ) ) {
                  result[0] += -0.009941254708853728;
                } else {
                  result[0] += -0.10119964001436527;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
              result[0] += 0.0539740244369791;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.03580465539747655;
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.03203662844576007;
                    } else {
                      result[0] += -0.051242889058505436;
                    }
                  }
                } else {
                  result[0] += 0.029892642347776416;
                }
              } else {
                result[0] += -0.05413315094718305;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
              result[0] += 0.014412716857228664;
            } else {
              result[0] += 0.1232028108924541;
            }
          } else {
            result[0] += -0.01585846923168084;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
            result[0] += -0.02624597375462442;
          } else {
            result[0] += -0.09963750314480446;
          }
        }
      }
    } else {
      result[0] += -0.08950558125603268;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
          result[0] += 0.0024769210139944297;
        } else {
          result[0] += -0.0278004429084193;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
          result[0] += -0.023887488532433887;
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0001905049065472189;
            } else {
              result[0] += -0.0922153505138883;
            }
          } else {
            result[0] += -0.10042415903628961;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
        if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0011946894557405495;
            } else {
              result[0] += 0.061440064881844184;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
              result[0] += -0.009710436209956859;
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.09719738915794818;
              } else {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.04801664677176137;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
                    result[0] += -0.0787175687658341;
                  } else {
                    result[0] += 0.08577073120218631;
                  }
                }
              }
            }
          }
        } else {
          result[0] += -0.06166508252601445;
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.09553252725543604;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.00583349393854378;
            } else {
              result[0] += -0.04963027557830743;
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                result[0] += -0.021142215273402654;
              } else {
                result[0] += -0.09149690717852821;
              }
            } else {
              result[0] += -0.04577519499581589;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.03849744796753107) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                    result[0] += 0.03182819593279132;
                  } else {
                    result[0] += -0.09681940657312839;
                  }
                } else {
                  result[0] += -0.0316708420514283;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.03236141402425835;
                  } else {
                    result[0] += -0.042775505661000746;
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.038860205347346015;
                  } else {
                    result[0] += 0.08172377807217279;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.03425345966035883;
              } else {
                result[0] += -0.05694740079839733;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.011314406759366397;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += 0.07272839662354891;
                } else {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.057120682430706726;
                  } else {
                    result[0] += -0.012992565775308837;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.016121694692898925;
                } else {
                  result[0] += 0.07840428494900441;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                    result[0] += -0.031699163436199655;
                  } else {
                    result[0] += 0.021190479692407425;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                    result[0] += 0.021490351693070227;
                  } else {
                    result[0] += -0.11073600986125037;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.04107969132241762;
                } else {
                  result[0] += -0.0912107495536375;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.0035624287364940535;
              } else {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10098505020141779) ) ) {
                    result[0] += 0.045758500285822054;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.014566824901477999;
                    } else {
                      result[0] += -0.06496233181879095;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.004881381988526279) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.08059478819646915;
                    } else {
                      result[0] += 0.04719398426590497;
                    }
                  } else {
                    result[0] += 0.06996793842682039;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.01525682245979353;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
                  result[0] += 0.0654367219592769;
                } else {
                  result[0] += -0.03297592267258564;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.09541855119586395;
                  } else {
                    result[0] += -0.02043056739935877;
                  }
                } else {
                  result[0] += 0.003009312701410476;
                }
              } else {
                result[0] += -0.10893287114595958;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.01670581727862396;
            } else {
              result[0] += -0.07148988286510169;
            }
          } else {
            result[0] += 0.02408649295384706;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.396947860717774326) ) ) {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.08719803272933645;
            } else {
              result[0] += -0.03430291209072162;
            }
          } else {
            result[0] += -0.08901570598957215;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
        result[0] += -0.03814550300792145;
      } else {
        result[0] += -0.10262475282340638;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.016029517439630575;
              } else {
                result[0] += -0.0731663390792887;
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.05741203378951562;
                  } else {
                    result[0] += 0.061472057224159674;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.0454749651697696;
                  } else {
                    result[0] += -0.09238581938155045;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
                      result[0] += -0.04475946903476504;
                    } else {
                      if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.051397842564827004;
                      } else {
                        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                          result[0] += 0.06708295919463493;
                        } else {
                          result[0] += -0.007853445862853236;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.031581900207590684;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.432135581970215732) ) ) {
                      result[0] += 0.09772008727587927;
                    } else {
                      result[0] += 0.004854823667542961;
                    }
                  } else {
                    result[0] += -0.09275562611307342;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.025369020687051415;
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
                  result[0] += -0.041114167158303655;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.08342753137778638;
                    } else {
                      result[0] += 0.045312722595186865;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.050118430328048295;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += -0.11535960942803336;
                      } else {
                        result[0] += 0.03238986563527563;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.883084774017335761) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += -0.06472269811612612;
                  } else {
                    result[0] += 0.024933176215125058;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.019401193371903207;
                    } else {
                      result[0] += -0.08896461807589567;
                    }
                  } else {
                    result[0] += -0.08339274093533332;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.041580997217096974;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.04358661996717386;
              } else {
                result[0] += -0.06756537884396802;
              }
            }
          } else {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.05649143327099836;
                } else {
                  result[0] += 0.010916857150543562;
                }
              } else {
                result[0] += -0.06915327494619647;
              }
            } else {
              result[0] += -0.08985821919188983;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
            result[0] += -0.051061780386173955;
          } else {
            result[0] += -0.012415181939417139;
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.1107442447821825;
          } else {
            result[0] += -0.060880531515210834;
          }
        }
      }
    } else {
      result[0] += -0.09725335420410787;
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0019344740349070263;
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.024190170368008024;
              } else {
                result[0] += -0.05208731915549543;
              }
            }
          } else {
            result[0] += 0.028581163502562526;
          }
        } else {
          if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.026591721986116874;
            } else {
              result[0] += 0.03733550952843239;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.05025428800951964;
              } else {
                result[0] += -0.014413990678351014;
              }
            } else {
              result[0] += 0.06440605603234333;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
              if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.02965620500359194;
              } else {
                result[0] += -0.03725293797370407;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += -0.033487621976895705;
                } else {
                  result[0] += 0.04982554710251866;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.034865283060578144;
                } else {
                  result[0] += -0.10674404018313842;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)140.5000000000000284) ) ) {
              result[0] += 0.042245719863818455;
            } else {
              result[0] += -0.015015104888127998;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
              result[0] += -0.064299758897759;
            } else {
              result[0] += 0.028270237999796468;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.10088692689309164;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.0492187727320334;
                } else {
                  result[0] += 4.21115282418831e-05;
                }
              }
            } else {
              result[0] += -0.11223570429663242;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
        result[0] += -0.034525536779271095;
      } else {
        result[0] += -0.10004324730769461;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)13.50000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.08886119557356165;
              } else {
                result[0] += 0.03128994942245365;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.07373559052251075;
              } else {
                if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.09417054806504488;
                } else {
                  result[0] += 0.06373123792266334;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.04647024580104376;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.011288373147227606;
              } else {
                result[0] += -0.02836664884690465;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
              result[0] += -0.04819103081959799;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.612332344055176669) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.12065915170952934;
                } else {
                  result[0] += 0.08048675944378643;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.06400904085952612;
                } else {
                  result[0] += -0.09010055727883488;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.90263271331787287) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.053050994873047763) ) ) {
                result[0] += 0.02842757353436271;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07653569374874795;
                  } else {
                    result[0] += 0.0336067820127094;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.031169315992986463;
                    } else {
                      result[0] += -0.1024427575422837;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                      result[0] += -0.11390020993595323;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.030375330037371768;
                      } else {
                        result[0] += -0.08384326212126397;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.06429767263309889;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
          result[0] += -0.00883234384766949;
        } else {
          result[0] += -0.08404311542960854;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.08637879965921683;
            } else {
              result[0] += -0.0008873673035883263;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.238486170768738237) ) ) {
              result[0] += 0.010350131119056914;
            } else {
              result[0] += -0.0999039575547046;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.559112548828125888) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.010559291354884747;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77165889739990412) ) ) {
                  result[0] += -0.013423532883916695;
                } else {
                  result[0] += -0.0957314314319736;
                }
              }
            } else {
              result[0] += 0.0020773888476859257;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0789882278190073;
            } else {
              result[0] += -0.012620089407993487;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.09927579357950334;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.222574234008789951) ) ) {
            result[0] += -0.04037125453114673;
          } else {
            result[0] += -0.07336452452848344;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)233.5000000000000284) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.689592361450196201) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                    result[0] += -0.0608849187297404;
                  } else {
                    result[0] += 0.0015861939314044022;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                    result[0] += -0.012681739367028516;
                  } else {
                    result[0] += 0.07334521438109355;
                  }
                }
              } else {
                result[0] += 0.047144486275881976;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.020952326944416153;
              } else {
                result[0] += -0.07849749171109446;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.011770774253642875;
            } else {
              result[0] += 0.05555083852755408;
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.016673095789629108;
              } else {
                result[0] += 0.04110336096077255;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.021073250497071032;
                  } else {
                    result[0] += -0.09788082105368348;
                  }
                } else {
                  result[0] += -0.002194103036877392;
                }
              } else {
                result[0] += -0.10484931042961154;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.01075156293715449;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                    result[0] += 0.018136723353185856;
                  } else {
                    result[0] += -0.10596083525841662;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.03492566247930048;
                } else {
                  result[0] += -0.0875255273344755;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.01225807954767703;
              } else {
                result[0] += 0.03960484827978462;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.013802969343429079;
        } else {
          result[0] += -0.08342251870063098;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
        result[0] += -0.03456274858527395;
      } else {
        result[0] += -0.09909900019495747;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11214685440063654) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.934722661972046787) ) ) {
                          result[0] += -0.009714869614425106;
                        } else {
                          result[0] += 0.06994038219709076;
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.014570632766361336;
                        } else {
                          result[0] += 0.09414425309047775;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.087577104568482333) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                          result[0] += -0.004952943144523773;
                        } else {
                          result[0] += -0.10112989625383828;
                        }
                      } else {
                        result[0] += 0.03972684583682126;
                      }
                    }
                  } else {
                    result[0] += -0.000813406820449554;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                    result[0] += -0.12499024402321263;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
                      result[0] += 0.02232214657653539;
                    } else {
                      result[0] += -0.06060805711756442;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)156.5000000000000284) ) ) {
                    result[0] += -0.08011667805023248;
                  } else {
                    result[0] += -0.01710629291401793;
                  }
                } else {
                  result[0] += 0.02387464144476482;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
                  result[0] += 0.07195705910400156;
                } else {
                  result[0] += -0.07380215502344638;
                }
              } else {
                result[0] += -0.10393170141747246;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                result[0] += -0.006984687769308504;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                  result[0] += -0.11020383813863184;
                } else {
                  if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.09218501588060862;
                  } else {
                    result[0] += -0.01663490372961626;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                  result[0] += 0.052912901141062146;
                } else {
                  result[0] += -0.10485929697976644;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.537947177886963779) ) ) {
                    result[0] += -0.08630488100259809;
                  } else {
                    result[0] += 0.02082521099640676;
                  }
                } else {
                  result[0] += -0.1095370015603363;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.338887453079224521) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
                result[0] += 0.09149239241531411;
              } else {
                result[0] += 0.01728504787578994;
              }
            } else {
              result[0] += -0.10495825608748083;
            }
          } else {
            result[0] += -0.10622547251151447;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
          result[0] += -0.017767784702452712;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.10702925789496878;
          } else {
            result[0] += -0.0517627527458778;
          }
        }
      }
    } else {
      result[0] += -0.09248486210420301;
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                result[0] += 0.04618222385401167;
              } else {
                result[0] += 0.00534432584771724;
              }
            } else {
              result[0] += -0.027074806991635825;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += 0.05958152276021004;
                } else {
                  result[0] += -0.04883322368542808;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.06841449996161879;
                } else {
                  result[0] += 0.02976352874333593;
                }
              }
            } else {
              result[0] += -0.0774140408282676;
            }
          }
        } else {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
                if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0063446994052860405;
                } else {
                  result[0] += -0.0484008676518389;
                }
              } else {
                result[0] += 0.05643730029343169;
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.027484116349704237;
                } else {
                  result[0] += 0.04831113097570797;
                }
              } else {
                result[0] += 0.07269511147829727;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                    result[0] += -0.05015304254361376;
                  } else {
                    result[0] += 0.01144983817329943;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                    result[0] += 0.017636166223077832;
                  } else {
                    result[0] += -0.10231523866134568;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.037184428706219234;
                } else {
                  result[0] += -0.09532987519828859;
                }
              }
            } else {
              result[0] += 0.03612244251524593;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.05840967815384349;
          } else {
            result[0] += -0.0036548443455983676;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.02721460993915836;
              } else {
                result[0] += -0.029309010794554915;
              }
            } else {
              result[0] += 0.04291408609979703;
            }
          } else {
            result[0] += -0.08449291399798203;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
        result[0] += -0.030266872647970323;
      } else {
        result[0] += -0.09675149244952433;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.182021141052246982) ) ) {
            result[0] += -0.009229741087363366;
          } else {
            result[0] += -0.09014812480717467;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                  if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.05515718808758144;
                  } else {
                    result[0] += 0.017885580885340032;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07419689762147699;
                  } else {
                    result[0] += -0.002494052375553715;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                    result[0] += 0.060193245768913065;
                  } else {
                    result[0] += -0.024897915737677544;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                    result[0] += -0.14410722588122124;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.020669987888880803;
                    } else {
                      result[0] += -0.09989231361136626;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.012702209292579493;
                  } else {
                    result[0] += -0.09688129645492165;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
                    result[0] += 0.01622514761925676;
                  } else {
                    result[0] += -0.08956223407985721;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                  result[0] += 0.04000236766066681;
                } else {
                  result[0] += -0.023541861663823906;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                result[0] += 0.06118610340152525;
              } else {
                result[0] += -0.012621177371040264;
              }
            } else {
              result[0] += -0.10098581386678665;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            result[0] += 0.0075994106792400765;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03861396989770834;
              } else {
                result[0] += 0.019400347370908344;
              }
            } else {
              result[0] += -0.07890141194617555;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
            result[0] += 0.012488195081916943;
          } else {
            result[0] += -0.09633508421297349;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
        result[0] += -0.02012402392005696;
      } else {
        if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.08114434870301793;
        } else {
          result[0] += -0.04200471752463907;
        }
      }
    }
  } else {
    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.280697107315064365) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                result[0] += 0.006293150247466232;
              } else {
                result[0] += -0.11010208523610794;
              }
            } else {
              result[0] += 0.03215897995652099;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += 0.03737193131704788;
            } else {
              result[0] += -0.0493323353200443;
            }
          }
        } else {
          result[0] += -0.04649349691421171;
        }
      } else {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.04301029001921261;
          } else {
            result[0] += -0.012692720729087557;
          }
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.03586177548037675;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += -0.01865004754030272;
            } else {
              result[0] += 0.07491695759465321;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)121.5000000000000142) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.422362327575684482) ) ) {
                    result[0] += -0.07182610470831782;
                  } else {
                    result[0] += 0.0676001060445267;
                  }
                } else {
                  result[0] += 0.013745616049888302;
                }
              } else {
                result[0] += 0.03292768381391605;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                  result[0] += 0.05527335291445765;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.02515251662610264;
                  } else {
                    result[0] += 0.05400422448053205;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                  result[0] += 0.03970941832538599;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.003238000652226364;
                  } else {
                    result[0] += -0.08686516839350902;
                  }
                }
              }
            }
          } else {
            result[0] += -0.02228829672949443;
          }
        } else {
          result[0] += -0.0281813261082499;
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)115.5000000000000142) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += -0.07989328294919269;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.048952843863461055;
                } else {
                  result[0] += -0.03294127599061684;
                }
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.021053728469299732;
              } else {
                result[0] += 0.08634972202052477;
              }
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.08734178251109107;
            } else {
              result[0] += -0.013094002410901018;
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.08641472768267217;
            } else {
              result[0] += -0.008884532421750874;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += -0.06048298230224525;
            } else {
              result[0] += -0.10740762125481332;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.0066129553356002214;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.21515866029596273;
                } else {
                  result[0] += -0.09276629072224335;
                }
              }
            } else {
              if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += -0.019824445850204226;
              } else {
                result[0] += 0.004025897903614026;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.607985973358155185) ) ) {
                result[0] += 0.06836925248857022;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.352615833282471591) ) ) {
                  result[0] += 0.016142018512597195;
                } else {
                  result[0] += -0.10312060572206834;
                }
              }
            } else {
              result[0] += -0.10185853740912257;
            }
          }
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
              result[0] += 0.009578276466805577;
            } else {
              result[0] += -0.0340101666174071;
            }
          } else {
            result[0] += -0.055346923614171074;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += -0.013595699623263714;
        } else {
          if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04564030691035786;
            } else {
              result[0] += -0.09153355870250765;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.04240206699928552;
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.047454941526071145;
              } else {
                result[0] += 0.10392744462784725;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.08858025883946105;
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  result[0] += 0.012685313809615423;
                } else {
                  result[0] += -0.030671731953973127;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                  result[0] += 0.01919608313495365;
                } else {
                  result[0] += 0.058320461936666824;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.016456106652393148;
              } else {
                result[0] += -0.05325800839359654;
              }
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.04385128533015098;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                    result[0] += 0.029944700190632653;
                  } else {
                    result[0] += -0.07105014048099215;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.95229363441467374) ) ) {
                    result[0] += -0.112706763335434;
                  } else {
                    result[0] += 0.07172837166677333;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += 0.05939155196104136;
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.040371296474565235;
                  } else {
                    result[0] += -0.02753433207964988;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += 0.011984138878957808;
                } else {
                  result[0] += 0.07240784494683102;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.006874401457213123;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                    result[0] += 0.01019453823015264;
                  } else {
                    result[0] += -0.08272538929537537;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                  result[0] += 0.02248945335088485;
                } else {
                  result[0] += -0.08714331907865362;
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.975242614746095526) ) ) {
                  result[0] += 0.005427357309906941;
                } else {
                  result[0] += -0.10019528288116866;
                }
              } else {
                if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77165889739990412) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                        result[0] += 0.039604174354078106;
                      } else {
                        result[0] += -0.03798092597574921;
                      }
                    } else {
                      result[0] += 0.04000562332162937;
                    }
                  } else {
                    result[0] += -0.013581030960075659;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.256982564926148349) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.07500994724961416;
                    } else {
                      result[0] += 0.0441342823786355;
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.009285140720330622;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.344720840454102451) ) ) {
                        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += 0.03860601451034254;
                        } else {
                          result[0] += -0.09671153655121772;
                        }
                      } else {
                        result[0] += 0.06746744047157736;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03089845294891517;
              } else {
                result[0] += -0.009498623067408285;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += -0.07186535060762866;
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.047696700670986435;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.03658282291794309;
                    } else {
                      result[0] += 0.029722662521970197;
                    }
                  }
                }
              } else {
                result[0] += -0.10632291894514434;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
          result[0] += 0.010265141056158361;
        } else {
          result[0] += -0.036710364591063915;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
        result[0] += -0.026192040210844366;
      } else {
        result[0] += -0.09305271030305803;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.90263271331787287) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.0032611190599900076;
              } else {
                result[0] += 0.02882413779069888;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
                result[0] += 0.012230694882492012;
              } else {
                result[0] += -0.06907587000120811;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
                  result[0] += 0.009080308177806811;
                } else {
                  result[0] += -0.0782814406485453;
                }
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)196.5000000000000284) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.00313213379910947;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)13.50000000000000178) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.08842499283331497;
                        } else {
                          result[0] += 0.028412368417197117;
                        }
                      } else {
                        result[0] += 0.040761111723721925;
                      }
                    } else {
                      result[0] += -0.02905654594093203;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.04271078635012311;
                  } else {
                    result[0] += -0.08223626571101861;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.23832273483276456) ) ) {
                result[0] += 0.07623095465754291;
              } else {
                result[0] += -0.07836997473409425;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39368534088134943) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.03785136721759861;
              } else {
                result[0] += -0.038502784579187294;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += -0.11708622380870592;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.05732801428676479;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.08789074642040483;
                  } else {
                    result[0] += 0.05642401989648969;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.014605907853542708;
              } else {
                result[0] += -0.07897589457353371;
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.07566629232611907;
              } else {
                result[0] += -0.016340221581988564;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.009262951279507303;
            } else {
              result[0] += -0.11043653738275412;
            }
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.75096368789673029) ) ) {
                result[0] += -0.07585026930753164;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.09383112124497711;
                } else {
                  result[0] += 0.10241620590535222;
                }
              }
            } else {
              result[0] += -0.015626540741675975;
            }
          }
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.10278338622836838;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                result[0] += -0.011749624383949378;
              } else {
                result[0] += -0.053228637964759906;
              }
            } else {
              result[0] += -0.07667501137680512;
            }
          }
        }
      }
    } else {
      result[0] += -0.08600364047841677;
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
                  if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.03012034506351285;
                  } else {
                    result[0] += -0.01659458021982688;
                  }
                } else {
                  result[0] += -0.09840719076594559;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)218.5000000000000284) ) ) {
                    result[0] += 0.0026594578863627493;
                  } else {
                    result[0] += 0.05140302794759312;
                  }
                } else {
                  result[0] += 0.05582091200816328;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.019323953029789044;
              } else {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                  result[0] += -0.06317391936514409;
                } else {
                  result[0] += -0.0015869353303134652;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.011117427510621477;
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += 0.055234768327209055;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.0810177385684967;
                  } else {
                    result[0] += -0.00019329293792504077;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.03432977267462209;
                  } else {
                    result[0] += -0.09603168742000162;
                  }
                } else {
                  result[0] += 0.06407486985107441;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)185.5000000000000284) ) ) {
            result[0] += 0.01848546871265991;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.940638065338136542) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                result[0] += -0.04581321766233593;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                  result[0] += 0.0253644317671198;
                } else {
                  result[0] += -0.02697586967066309;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.02121583623050625;
              } else {
                result[0] += -0.07414887126748866;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.01017184207070073;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
            result[0] += 0.0420875780159066;
          } else {
            result[0] += -0.10255909768154664;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
        result[0] += -0.029318848824538647;
      } else {
        result[0] += -0.09464741672278738;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
            result[0] += -0.012333449611068123;
          } else {
            result[0] += -0.08496194734528718;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                  result[0] += 0.033666464516139326;
                } else {
                  result[0] += -0.035922802485847644;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                    result[0] += 0.05042150620382524;
                  } else {
                    result[0] += -0.026945776776457604;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                    result[0] += -0.08912219812316746;
                  } else {
                    result[0] += -0.015486343039544213;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.632002353668214667) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.008600422934706267;
                  } else {
                    result[0] += -0.09172456732145927;
                  }
                } else {
                  result[0] += 0.0018278702220452599;
                }
              } else {
                result[0] += 0.021765367638280964;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                    result[0] += 0.025259336702159227;
                  } else {
                    result[0] += -0.10529577826225063;
                  }
                } else {
                  result[0] += -0.09599737850865536;
                }
              } else {
                result[0] += 0.05661005273213931;
              }
            } else {
              result[0] += -0.0967388810508924;
            }
          }
        }
      } else {
        result[0] += -0.05462008643310611;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
        if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.024007428244280322;
          } else {
            result[0] += -0.013287652441860982;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
            result[0] += 0.01015076733408244;
          } else {
            result[0] += -0.0884854994529314;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += -0.032109472526917225;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              result[0] += 0.09465258736568176;
            } else {
              result[0] += -0.09644492841403443;
            }
          }
        } else {
          result[0] += -0.0824205791776811;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  result[0] += 0.01793069437747055;
                } else {
                  result[0] += -0.022204372057734283;
                }
              } else {
                result[0] += -0.03922358940715251;
              }
            } else {
              result[0] += 0.027327538728334846;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                result[0] += 0.04580541584204336;
              } else {
                result[0] += -0.038904349572416105;
              }
            } else {
              if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.03209212240311201;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.027058570826387313;
                  } else {
                    result[0] += -0.08380118266123908;
                  }
                } else {
                  result[0] += 0.058720563575932055;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.109245061874390537) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0008548511281103698;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                    result[0] += 0.007444398902434481;
                  } else {
                    result[0] += -0.08128862223370947;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.890260934829712802) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07112730717905309;
                    } else {
                      result[0] += 0.022100333698179587;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.874179124832154208) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.02943945706217521;
                      } else {
                        result[0] += -0.07118972114545136;
                      }
                    } else {
                      result[0] += 0.04583782564284361;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += 0.05350193214158654;
                  } else {
                    result[0] += -0.09084136921865407;
                  }
                }
              }
            } else {
              result[0] += 0.02743044847520373;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.447260618209839755) ) ) {
                  result[0] += 0.05199927136144941;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    result[0] += 0.03905783043689812;
                  } else {
                    result[0] += -0.039836658733501645;
                  }
                }
              } else {
                result[0] += -0.020876858707658222;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                    result[0] += 0.012504029750034649;
                  } else {
                    result[0] += -0.07081721342375795;
                  }
                } else {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                      result[0] += -0.06193744070783072;
                    } else {
                      result[0] += 0.027695264672548955;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.08168586500938335;
                    } else {
                      result[0] += 0.017981134370083498;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.11108795273134896;
                } else {
                  result[0] += -0.03537727730108501;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
          result[0] += 0.008780178447333555;
        } else {
          result[0] += -0.033038659076044495;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
        result[0] += -0.02655730074650377;
      } else {
        result[0] += -0.09283714802333702;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                result[0] += -0.08031772770511975;
              } else {
                if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.025045791323284012;
                } else {
                  result[0] += -0.0731003078708165;
                }
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.036080170210902096;
                } else {
                  result[0] += -0.08586654673260899;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)13.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1256904602050799) ) ) {
                    result[0] += 0.010161882549449178;
                  } else {
                    result[0] += 0.07896007767526887;
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += -0.07093882408166176;
                    } else {
                      result[0] += 0.01859778394602387;
                    }
                  } else {
                    result[0] += 0.022660685459236617;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.920663833618164951) ) ) {
                result[0] += 0.0284714111266461;
              } else {
                result[0] += -0.06333661390742695;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.10319108931674643;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                  result[0] += -0.09879423844707047;
                } else {
                  result[0] += 0.005871856651503828;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.03943029982933438;
              } else {
                result[0] += 0.026305191700672984;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
                    result[0] += -0.08190653828877678;
                  } else {
                    result[0] += 0.054818623323040494;
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0008412328774219411;
                  } else {
                    result[0] += -0.07209347503262942;
                  }
                }
              } else {
                result[0] += -0.0727362155915433;
              }
            }
          } else {
            result[0] += 0.02752311102042227;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
          result[0] += 0.0018895696166825002;
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.02888611704823122;
          } else {
            result[0] += -0.09832678560652111;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)131.5000000000000284) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  result[0] += -0.06013865860230803;
                } else {
                  result[0] += 0.0005734317460923927;
                }
              } else {
                result[0] += 0.010825023133138547;
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.06084648265603974;
              } else {
                result[0] += 0.02300587977946487;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.400641441345215288) ) ) {
              result[0] += 0.01928995386936933;
            } else {
              result[0] += -0.08843631898659936;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
            result[0] += -0.005824561437362457;
          } else {
            if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += -0.06420496751371371;
            } else {
              if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.0004045153347817715;
              } else {
                result[0] += -0.04762434111199255;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.08939381511239569;
        } else {
          result[0] += -0.045250766619669786;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.729812622070313388) ) ) {
                result[0] += -0.07127798526409956;
              } else {
                result[0] += 0.013504985525715822;
              }
            } else {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.02264217647001067;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.601370334625245029) ) ) {
                  result[0] += -0.01931148379090855;
                } else {
                  result[0] += 0.06245055448240794;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.32530093193054288) ) ) {
                result[0] += 0.03866724500535773;
              } else {
                result[0] += -0.03441707294308232;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.689592361450196201) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09427356719970881) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.99033999443054288) ) ) {
                      result[0] += 0.0027784237268174586;
                    } else {
                      result[0] += 0.04853077378770758;
                    }
                  } else {
                    result[0] += -0.045381704840443876;
                  }
                } else {
                  result[0] += 0.04449918482944263;
                }
              } else {
                result[0] += 0.06974967782804795;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                result[0] += -0.04713782161088004;
              } else {
                result[0] += 0.017115288488575863;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.09326739373976427;
                  } else {
                    result[0] += -0.02724112328635189;
                  }
                } else {
                  result[0] += -0.0020879120547454607;
                }
              } else {
                result[0] += -0.09891873563645248;
              }
            }
          } else {
            result[0] += 0.01587182157076301;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += -0.008332947415370773;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
            result[0] += 0.03959571789231451;
          } else {
            result[0] += -0.1001538587858847;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        result[0] += -0.07955733302067815;
      } else {
        result[0] += -0.0013489112472452904;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
              result[0] += -0.004399183221802561;
            } else {
              result[0] += -0.050994521896530365;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.07318513668531369;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)17.50000000000000355) ) ) {
                    result[0] += -0.04901414375518692;
                  } else {
                    result[0] += 0.05081130224876049;
                  }
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.874179124832154208) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                      if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += -0.09398164647748795;
                      } else {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                          result[0] += -0.03397669146269081;
                        } else {
                          result[0] += 0.030102369136968916;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.03849744796753107) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                              result[0] += -0.03360122548695485;
                            } else {
                              result[0] += 0.0714640404058711;
                            }
                          } else {
                            result[0] += -0.0831135531501171;
                          }
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                              result[0] += 0.0028848058679432588;
                            } else {
                              result[0] += 0.08087420422861215;
                            }
                          } else {
                            result[0] += 0.01699437267896797;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.559112548828125888) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.12533096364098315;
                          } else {
                            result[0] += 0.007583280887291462;
                          }
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.036670446395874912) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.003935290714590523;
                              } else {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.177185058593750444) ) ) {
                                  result[0] += 0.008078989383323047;
                                } else {
                                  result[0] += -0.09002894590833938;
                                }
                              }
                            } else {
                              if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += -0.025206280964879226;
                              } else {
                                result[0] += 0.0486459421836965;
                              }
                            }
                          } else {
                            result[0] += 0.04031938153672119;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.04993966072563881;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
                      result[0] += 0.04712687652285349;
                    } else {
                      result[0] += -0.09230258812605494;
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.10478371317943974;
                    } else {
                      result[0] += -0.0206380843509689;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
                result[0] += 0.01128342622378172;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.015945925252901946;
                  } else {
                    result[0] += 0.09749389723798596;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.0535642304584164;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                        result[0] += 0.018937675411592293;
                      } else {
                        result[0] += -0.06891036834661717;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                      result[0] += -0.024835610289849375;
                    } else {
                      result[0] += -0.08962983299803319;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.716979026794434482) ) ) {
              result[0] += 0.08325304414379782;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.352615833282471591) ) ) {
                result[0] += 0.01892015806799157;
              } else {
                result[0] += -0.08674219612916406;
              }
            }
          } else {
            result[0] += -0.09404960606472579;
          }
        }
      } else {
        result[0] += -0.06188509969576849;
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
        result[0] += -0.01972405941515365;
      } else {
        result[0] += -0.0649152656405372;
      }
    }
  } else {
    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
              result[0] += 0.03823882361341194;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.918693304061890537) ) ) {
                result[0] += 0.020246529589111398;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.05956441684927595;
                } else {
                  result[0] += 0.021535363372026127;
                }
              }
            }
          } else {
            result[0] += -0.027063645028631584;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.01077156498464895;
            } else {
              result[0] += -0.031747170669767226;
            }
          } else {
            result[0] += -0.07863707394539655;
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.924915313720704901) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
              result[0] += -0.01819711835501541;
            } else {
              result[0] += 0.03767277995844738;
            }
          } else {
            result[0] += 0.04535680839475243;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.655387401580811435) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.005703974245450867;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                  result[0] += 0.0026701047524906136;
                } else {
                  result[0] += -0.0679258450920499;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.025506849855157665;
              } else {
                result[0] += -0.09038558028160243;
              }
            }
          } else {
            result[0] += 0.02605770384725898;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
        result[0] += -0.04052223512260474;
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += 0.027619669595529635;
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)75.50000000000001421) ) ) {
            result[0] += 0.025797497592233612;
          } else {
            result[0] += -0.06668150036119389;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          result[0] += 0.0001236428105855176;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
              result[0] += 0.2458685125469108;
            } else {
              result[0] += -0.0947110111508966;
            }
          } else {
            result[0] += 0.021507005717839898;
          }
        }
      } else {
        result[0] += -0.058134733374137416;
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.75874996185302912) ) ) {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.045251778126769115;
                } else {
                  result[0] += 0.0052163111045181254;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                  result[0] += -0.09418202105358148;
                } else {
                  result[0] += 0.04886577838645036;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.026272822781698874;
                } else {
                  result[0] += -0.05990328319925858;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0345899491225428;
                } else {
                  result[0] += -0.028855349506688072;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.238486170768738237) ) ) {
              result[0] += 0.0211787809491953;
            } else {
              result[0] += -0.08482614866815291;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.559112548828125888) ) ) {
            result[0] += -0.009193176995911137;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.10261304045508451;
              } else {
                result[0] += -0.05014539750649919;
              }
            } else {
              result[0] += 0.0039418234085240225;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          result[0] += -0.08455886678094165;
        } else {
          result[0] += -0.04104339586955376;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0019939533947580953;
              } else {
                result[0] += 0.030953976419962122;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
                result[0] += -0.05282586961359145;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                  result[0] += -0.06102405927701739;
                } else {
                  result[0] += 0.061728732653414525;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                result[0] += 0.04202272472360076;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.010033503112198661;
                } else {
                  result[0] += -0.06279672701174578;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.030585954585228614;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                  result[0] += 0.0059170264584853735;
                } else {
                  result[0] += 0.06398174875510984;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.01636789950954679;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.473471879959107333) ) ) {
                    if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.022237497486734564;
                    } else {
                      result[0] += -0.07480396661312014;
                    }
                  } else {
                    result[0] += 0.04505886963885656;
                  }
                }
              } else {
                result[0] += 0.021068189226123232;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.04105374750356323;
                  } else {
                    result[0] += -0.023073125294120755;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.006788545966817484;
                      } else {
                        result[0] += -0.06879895070540261;
                      }
                    } else {
                      result[0] += 0.05139912333667388;
                    }
                  } else {
                    result[0] += -0.08957197065954958;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.01856720317166753;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                    result[0] += 0.04444129569774178;
                  } else {
                    result[0] += -0.09652261028937614;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.036242591331515396;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                    result[0] += -0.07730455741769665;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.026057833281036048;
                    } else {
                      result[0] += -0.031140788603219706;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.09339814495627502;
                  } else {
                    if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.05708732783668807;
                      } else {
                        result[0] += -0.027659858374042015;
                      }
                    } else {
                      result[0] += 0.06954026294275832;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.07296253926349665;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          result[0] += -0.03733578435963991;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.025125604652716904;
              } else {
                result[0] += -0.024188953982764192;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.039331857231799806;
              } else {
                result[0] += 0.04931253808368702;
              }
            }
          } else {
            result[0] += -0.07523789840808753;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
        result[0] += -0.015074538310703484;
      } else {
        result[0] += -0.08613889950583524;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += 0.0010460295533251517;
            } else {
              result[0] += -0.04790131631893829;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
                result[0] += 0.05564017322817486;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.352615833282471591) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += 0.009964878778602583;
                    } else {
                      result[0] += -0.08811891482919204;
                    }
                  } else {
                    result[0] += 0.03774317811095308;
                  }
                } else {
                  result[0] += -0.0972812824652024;
                }
              }
            } else {
              result[0] += -0.09073663698542321;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += 0.01782879285138492;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.009914830215701803;
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
                    result[0] += -0.08768040737142543;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.28931427001953303) ) ) {
                      result[0] += -0.0610822222446622;
                    } else {
                      result[0] += 0.1450029652950722;
                    }
                  }
                }
              } else {
                result[0] += -0.05686237343127142;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.03495035733368713;
            } else {
              result[0] += -0.08810799593421498;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
          result[0] += -0.012610670619872442;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.09698553525036065;
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                result[0] += -0.006417149870580917;
              } else {
                result[0] += -0.044394261682404075;
              }
            } else {
              result[0] += -0.06601146517987526;
            }
          }
        }
      }
    } else {
      result[0] += -0.0761241137153537;
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
                result[0] += 0.03628717504635745;
              } else {
                result[0] += 0.009889974691078052;
              }
            } else {
              result[0] += -0.02184130987890806;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
                  result[0] += 0.010054221973978787;
                } else {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.02856296657902256;
                  } else {
                    result[0] += -0.0964604809617523;
                  }
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02509248920814406;
                } else {
                  result[0] += -0.014038269056680841;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.10260374427609781;
              } else {
                result[0] += -0.04023537362576951;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.350515365600586826) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.623641014099121982) ) ) {
                  result[0] += -0.06941893894967062;
                } else {
                  result[0] += 0.02613214936983444;
                }
              } else {
                if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.01729793831725432;
                } else {
                  result[0] += 0.025634927557663268;
                }
              }
            } else {
              result[0] += 0.043825354700373395;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.03021296950411232;
                } else {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.023318532241613735;
                      } else {
                        result[0] += -0.026324593752907962;
                      }
                    } else {
                      result[0] += 0.04031949957269481;
                    }
                  } else {
                    result[0] += 0.001233644882457742;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                      result[0] += 0.012805433260345021;
                    } else {
                      result[0] += 0.06887489660009875;
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.007324400749808022;
                      } else {
                        result[0] += -0.09667502726582511;
                      }
                    } else {
                      result[0] += 0.025376662180459777;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)211.5000000000000284) ) ) {
                      result[0] += 0.057674195119729425;
                    } else {
                      result[0] += -0.04058422734546008;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.05647281813759108;
                    } else {
                      result[0] += 0.05018872980192031;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                result[0] += 0.01892660256229134;
              } else {
                result[0] += -0.08051813438889176;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
          result[0] += -0.03529344971786549;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.352615833282471591) ) ) {
              result[0] += 0.015409890844330781;
            } else {
              result[0] += 0.048791249759335714;
            }
          } else {
            if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0808844729806066;
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.03650517537103248;
              } else {
                result[0] += -0.037297319067191935;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
          result[0] += -0.02130467357814142;
        } else {
          result[0] += -0.09443883404082166;
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
          result[0] += 0.04170679539024974;
        } else {
          result[0] += -0.06007004418534987;
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)68.50000000000001421) ) ) {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += -0.0049087552462532885;
        } else {
          if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.20424372829626353;
          } else {
            result[0] += -0.07310204629900122;
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)48.50000000000000711) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)16.50000000000000355) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                    result[0] += -0.10178217956684399;
                  } else {
                    result[0] += 0.03418357619233416;
                  }
                } else {
                  result[0] += -0.005572989057433683;
                }
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.06969621034209277;
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.05902901558651405;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.08292265734195303;
                    } else {
                      result[0] += 0.07418047454542107;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.037131056112690825;
            }
          } else {
            result[0] += -0.0008343128493838755;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.90263271331787287) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
                  result[0] += -0.040915872400449686;
                } else {
                  result[0] += 0.014907653928474544;
                }
              } else {
                result[0] += 0.02660730778827852;
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
                result[0] += 0.016529031003028858;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.549646615982056552) ) ) {
                  result[0] += 0.0582951723435827;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.031724496691088344;
                  } else {
                    result[0] += -0.09818522071051766;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
              result[0] += -0.007627481116306789;
            } else {
              result[0] += -0.07845234185521249;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
        result[0] += -0.009201055337232655;
      } else {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)212.5000000000000284) ) ) {
            result[0] += -0.03764582861475546;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.009596350210006425;
            } else {
              result[0] += -0.06265770811997316;
            }
          }
        } else {
          result[0] += -0.0739078709465486;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.03082169759981301;
            } else {
              result[0] += -0.04412124122670853;
            }
          } else {
            if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
                result[0] += 0.015594637490694894;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += 0.0030402802539414115;
                } else {
                  result[0] += -0.06638374791733331;
                }
              }
            } else {
              result[0] += 0.008680339343160239;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.555368185043335849) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.623641014099121982) ) ) {
                  result[0] += -0.08036501314015286;
                } else {
                  result[0] += 0.03583187467091242;
                }
              } else {
                result[0] += -0.011349093056591054;
              }
            } else {
              result[0] += 0.029211122006903647;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.023007432004746583;
                } else {
                  if ( UNLIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0010636560752967717;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                      result[0] += -0.022551495198115768;
                    } else {
                      result[0] += 0.027644664174390033;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += -0.05085903159139979;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)210.5000000000000284) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.14221357492931394;
                          } else {
                            result[0] += -0.05578993876053473;
                          }
                        } else {
                          result[0] += 0.07156612819385137;
                        }
                      } else {
                        result[0] += -0.007493265469962498;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.1268346084364391;
                        } else {
                          result[0] += 0.014677503215385988;
                        }
                      } else {
                        result[0] += 0.04665779366456818;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                        result[0] += -0.05962741573912637;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                          result[0] += 0.022079646827092594;
                        } else {
                          result[0] += 0.11684876259801165;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.658699750900269443) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.847910165786744052) ) ) {
                          result[0] += -0.04720705850234009;
                        } else {
                          result[0] += 0.04127558512815356;
                        }
                      } else {
                        result[0] += 0.06584711871851927;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += -0.06739674059270562;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.357462406158449042) ) ) {
          result[0] += -0.03508388183101626;
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)76.50000000000001421) ) ) {
              result[0] += 0.04696635177827957;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.023903640764091072;
              } else {
                result[0] += -0.01424333784747898;
              }
            }
          } else {
            result[0] += -0.07066465519880265;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
          result[0] += -0.019530241257826493;
        } else {
          result[0] += -0.0926138783037875;
        }
      } else {
        result[0] += 0.008958613121462785;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                      result[0] += -0.07580593689588316;
                    } else {
                      result[0] += 0.03411823445763459;
                    }
                  } else {
                    result[0] += 0.01764483203243589;
                  }
                } else {
                  result[0] += -0.07999134431561108;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                  result[0] += 0.0067678025726224545;
                } else {
                  result[0] += -0.08250334678818165;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                result[0] += 0.034392278627866396;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += 0.043499888351667994;
                } else {
                  if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.008243677674438015;
                  } else {
                    result[0] += -0.09513645473602074;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                  if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.012471820211103888;
                  } else {
                    result[0] += -0.096356146572597;
                  }
                } else {
                  if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.06249776600605137;
                    } else {
                      result[0] += 0.015529227840081768;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                        result[0] += -0.11520103935622515;
                      } else {
                        result[0] += 0.035899229545358645;
                      }
                    } else {
                      result[0] += 0.029903178212893358;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.006368627966971728;
                } else {
                  result[0] += -0.07147249946638325;
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                    result[0] += -0.12872914259150475;
                  } else {
                    result[0] += -0.035018145654554377;
                  }
                } else {
                  result[0] += -0.1018915796349229;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += 0.058870514549902254;
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                    result[0] += -0.022612506960019804;
                  } else {
                    result[0] += -0.09522738102903314;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)114.5000000000000142) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)112.5000000000000142) ) ) {
                result[0] += -0.03119343105515998;
              } else {
                result[0] += 0.037573564438202194;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.06187798633535178;
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.07471495436274467;
                } else {
                  result[0] += -0.010640549023332897;
                }
              }
            }
          } else {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.010066625763765375;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.02958359052005491;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      result[0] += 0.0287004207863957;
                    } else {
                      result[0] += -0.05723022576483579;
                    }
                  }
                } else {
                  result[0] += 0.04613085992402194;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0002735850361548793;
                } else {
                  result[0] += 0.030536845782860395;
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.01129188016820181;
                } else {
                  result[0] += -0.08443272596083323;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.635775566101075995) ) ) {
          result[0] += 0.003432453180070145;
        } else {
          result[0] += -0.08199565081865019;
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
        result[0] += -0.0189553363035487;
      } else {
        result[0] += -0.07891561106896235;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.019545435022908317;
          } else {
            result[0] += -0.07931146291150475;
          }
        } else {
          result[0] += -0.08435747863953623;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.007518229791653095;
          } else {
            result[0] += -0.039879567004022815;
          }
        } else {
          if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.012666613695896495;
          } else {
            result[0] += -0.03912434293382247;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.400641441345215288) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.10056790066873883;
          } else {
            result[0] += -0.04047066309400573;
          }
        } else {
          result[0] += -0.07276717276498838;
        }
      } else {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.07004703476719534;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += 0.02394998046998782;
                } else {
                  result[0] += -0.0620493697049325;
                }
              } else {
                result[0] += -0.07902828290852121;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.10363643067669004;
              } else {
                result[0] += 0.003291928522686302;
              }
            } else {
              if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.04950782627993855;
              } else {
                result[0] += -0.0016139304116279591;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.08897640543516604;
          } else {
            result[0] += -0.022170184277019393;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.03360784813613862;
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.009059809291303707;
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
                        result[0] += -0.048289205604509486;
                      } else {
                        result[0] += 0.090998678478685;
                      }
                    } else {
                      result[0] += -0.059049473897604836;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.10316099067976417;
                        } else {
                          result[0] += -0.044142794232271396;
                        }
                      } else {
                        result[0] += 0.00789172783465861;
                      }
                    } else {
                      result[0] += 0.02134059922873914;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += 0.023700381892369546;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.09051460546497425;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                          result[0] += -0.0594070463865966;
                        } else {
                          result[0] += 0.03923903521058348;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += 0.04671921774472733;
                } else {
                  result[0] += 0.003846179885984017;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.06973232713686921;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.222574234008789951) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.02232859550370926;
                      } else {
                        result[0] += -0.06482180811248361;
                      }
                    } else {
                      result[0] += 0.06376552588355981;
                    }
                  } else {
                    result[0] += -0.01513588368516524;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                      result[0] += 0.00883547999369984;
                    } else {
                      result[0] += -0.07835386123245724;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.02995371174816526;
                    } else {
                      result[0] += -0.019622766952988868;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += 0.03076807907017587;
                  } else {
                    result[0] += -0.07667366763422775;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              result[0] += 0.1294343859604524;
            } else {
              result[0] += -0.09288376790865217;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
              result[0] += 0.048291053351671505;
            } else {
              result[0] += -0.05484765927951212;
            }
          }
        }
      } else {
        result[0] += -0.06780471080157727;
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.006351377163347648;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
            result[0] += -0.008157158818933426;
          } else {
            result[0] += -0.050953991577506574;
          }
        }
      } else {
        result[0] += -0.054844095644623274;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  result[0] += 0.011497137926573612;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
                    result[0] += -0.03988371673448339;
                  } else {
                    result[0] += 0.017696338106327033;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.0038000491304641965;
                } else {
                  result[0] += -0.06198877361760019;
                }
              }
            } else {
              result[0] += 0.02188885558142086;
            }
          } else {
            if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.032670732929579095;
              } else {
                result[0] += -0.012229297174803727;
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  result[0] += 0.048018517574318416;
                } else {
                  result[0] += 0.002599808398308177;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.156140089035035068) ) ) {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0547784194093531;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                      result[0] += -0.09073167528417869;
                    } else {
                      result[0] += 0.08272513919465631;
                    }
                  }
                } else {
                  result[0] += 0.08302645460031637;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.013267562562676753;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.046974216325957226;
              } else {
                result[0] += -0.009332432263172456;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                    result[0] += -0.0791063762190129;
                  } else {
                    result[0] += -0.007206530336660215;
                  }
                } else {
                  result[0] += -0.09025124527364174;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
                  result[0] += 0.0213614289618387;
                } else {
                  result[0] += -0.022556803240899787;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
            result[0] += -0.047370433892746415;
          } else {
            result[0] += 0.013853986099741028;
          }
        } else {
          result[0] += -0.07551424462605562;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.852773189544679511) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.03589193455579639;
        } else {
          result[0] += 0.04298476922512512;
        }
      } else {
        result[0] += -0.0837544815754432;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.18063186896472738;
                } else {
                  result[0] += 0.0022375646810553046;
                }
              } else {
                result[0] += -0.017158021431986734;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.200417995452881748) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.189289569854737216) ) ) {
                  result[0] += 0.06364071752803174;
                } else {
                  result[0] += -0.03756279073980753;
                }
              } else {
                result[0] += -0.07211849986077572;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.010069938863827526;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.07762757357282671;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                    result[0] += -0.02583786531823087;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += 0.08247493998037318;
                    } else {
                      result[0] += -0.1088796479752136;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                  result[0] += 0.002324870289009081;
                } else {
                  result[0] += -0.04473171079091817;
                }
              } else {
                result[0] += -0.07933402797216825;
              }
            }
          }
        } else {
          result[0] += -0.04479927368674935;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
          result[0] += -0.014720309048930734;
        } else {
          result[0] += -0.09437900282340925;
        }
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        result[0] += -0.012090810146494626;
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
            result[0] += -0.014940926337404762;
          } else {
            result[0] += -0.045360373589202925;
          }
        } else {
          result[0] += -0.0706113897071352;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                    result[0] += -0.0037196725541305663;
                  } else {
                    result[0] += 0.029876385045822587;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
                    result[0] += -0.03705354215732448;
                  } else {
                    result[0] += 0.016135169353275997;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.432135581970215732) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += 0.027836629389078188;
                  } else {
                    result[0] += -0.05602444642974509;
                  }
                } else {
                  result[0] += 0.035996462019252745;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.022669126727390847;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.028481011297586678;
                } else {
                  result[0] += -0.0816548441484848;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                result[0] += 0.0481925348489597;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.024907998909000275;
                } else {
                  result[0] += -0.03561814102144858;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.029397520361381803;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.03556013033331549;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.382196187973023349) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.0386696519117196;
                      } else {
                        result[0] += -0.09736610371428561;
                      }
                    } else {
                      result[0] += 0.06703909462128602;
                    }
                  } else {
                    result[0] += 0.087806750900655;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.21334457397461115) ) ) {
              result[0] += 0.01436816834933839;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += -0.1005771355607936;
                  } else {
                    result[0] += -0.002043121955647011;
                  }
                } else {
                  result[0] += -0.06944427933351931;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                    result[0] += 0.052067890486810466;
                  } else {
                    result[0] += -0.017806942140609933;
                  }
                } else {
                  result[0] += -0.04010769530592012;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.043397122175981476;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.044354758068220715;
                } else {
                  result[0] += 0.012030232737406131;
                }
              }
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                    result[0] += -0.07664013851574099;
                  } else {
                    result[0] += -0.010126332281328677;
                  }
                } else {
                  result[0] += -0.08941807564129822;
                }
              } else {
                if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.06183158219922222;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += -0.07902663626093695;
                  } else {
                    result[0] += 0.006371353081853011;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.210240364074708808) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.0718277902065104;
            } else {
              result[0] += 0.010723430695764723;
            }
          } else {
            result[0] += 0.01263115196993554;
          }
        } else {
          result[0] += -0.078265821967131;
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.132412433624269354) ) ) {
          result[0] += -0.019749137434159555;
        } else {
          result[0] += -0.09237199103597266;
        }
      } else {
        result[0] += 0.013692666796184257;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
                      result[0] += 0.006017684038441498;
                    } else {
                      result[0] += 0.03291076708639979;
                    }
                  } else {
                    result[0] += -0.07351661021293784;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
                      result[0] += -0.023007608985778864;
                    } else {
                      result[0] += 0.04645648414682843;
                    }
                  } else {
                    result[0] += -0.07690099937590222;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.02300728238748249;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.663129329681397373) ) ) {
                      result[0] += 0.08385337160462993;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.763591527938843662) ) ) {
                        result[0] += 0.04273721832565132;
                      } else {
                        result[0] += -0.015464119486011339;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.23832273483276456) ) ) {
                    result[0] += 0.03822424794746973;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.010853444774887798;
                    } else {
                      result[0] += -0.09204905415217908;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
                      result[0] += -0.03661899035541752;
                    } else {
                      result[0] += 0.056801868934992034;
                    }
                  } else {
                    if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                          result[0] += 0.07139435501762645;
                        } else {
                          result[0] += 0.025151710696473463;
                        }
                      } else {
                        if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.033044738046539415;
                        } else {
                          result[0] += -0.028053400639306766;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                          result[0] += -0.10387457935419848;
                        } else {
                          result[0] += 0.028231057878616572;
                        }
                      } else {
                        result[0] += 0.021581508676753104;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                      result[0] += 0.009814254678909305;
                    } else {
                      result[0] += -0.05593863059762025;
                    }
                  } else {
                    result[0] += -0.06532833350166305;
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                      result[0] += -0.12614841382227557;
                    } else {
                      result[0] += -0.03284527899013338;
                    }
                  } else {
                    result[0] += -0.09748334493536148;
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                    result[0] += 0.05469966701172603;
                  } else {
                    result[0] += -0.0361635416767106;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)114.5000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.042003232552917075;
                } else {
                  result[0] += 0.03427955994260407;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += 0.051282438327309866;
                  } else {
                    result[0] += -0.0918588615887933;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.06805132096796501;
                  } else {
                    result[0] += -0.0018572896465861761;
                  }
                }
              }
            } else {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  result[0] += -0.008981976479126753;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.012712764182697492;
                  } else {
                    result[0] += 0.040829486856444865;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.0016688021598285894;
                  } else {
                    result[0] += 0.02665988263887055;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)211.5000000000000284) ) ) {
                      result[0] += 0.003091985364117157;
                    } else {
                      result[0] += -0.0334448015380296;
                    }
                  } else {
                    result[0] += -0.08090786634684388;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
            result[0] += 0.007837070665416836;
          } else {
            result[0] += -0.08066372383290762;
          }
        }
      } else {
        result[0] += -0.04764192357018227;
      }
    } else {
      result[0] += -0.020929702977094844;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += -0.0310114530437842;
        } else {
          result[0] += -0.07488020317763268;
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.09574428301496689;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
            result[0] += 0.0012504386301397863;
          } else {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.011167388939281233;
            } else {
              result[0] += -0.033322467132992487;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.0334147624958261;
          } else {
            result[0] += -0.05856343084666778;
          }
        } else {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.05655310558034339;
            } else {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.05478692481249575;
                } else {
                  result[0] += 0.026240156089678376;
                }
              } else {
                result[0] += -0.07340486400902933;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.04051521585474288;
            } else {
              result[0] += 0.03271301078984653;
            }
          }
        }
      } else {
        result[0] += -0.05091349702334113;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                    result[0] += 0.011667531851814903;
                  } else {
                    result[0] += -0.06813934279583753;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += 0.006458337017340191;
                  } else {
                    result[0] += -0.07157172290516678;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.020906862183765647;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.256982564926148349) ) ) {
                      result[0] += 0.06548095152324589;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                        result[0] += 0.029397830299954922;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.06952689698954342;
                        } else {
                          result[0] += 0.06424808764413248;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += 0.031695270986764355;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
                        result[0] += -0.07588661063778855;
                      } else {
                        result[0] += 0.01598213265944318;
                      }
                    } else {
                      result[0] += -0.08832623200490392;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
                      result[0] += -0.035927335085562284;
                    } else {
                      result[0] += 0.05268747814781355;
                    }
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.783749341964722568) ) ) {
                          result[0] += 0.0661345301120007;
                        } else {
                          result[0] += 0.021300370791855563;
                        }
                      } else {
                        result[0] += 0.009915600746158562;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                            result[0] += -0.08412423438384709;
                          } else {
                            result[0] += 0.0004762033953291062;
                          }
                        } else {
                          result[0] += 0.03022560854856566;
                        }
                      } else {
                        result[0] += 0.04048178162765874;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.008766433165458993;
                  } else {
                    result[0] += -0.06125216705662737;
                  }
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.18732333183288663) ) ) {
                      result[0] += -0.12214994887347652;
                    } else {
                      result[0] += -0.031197510118816947;
                    }
                  } else {
                    result[0] += -0.09501738613661723;
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                    result[0] += 0.048520073132689634;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)58.50000000000000711) ) ) {
                      result[0] += -0.01979204947320115;
                    } else {
                      result[0] += -0.08949316793916691;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)114.5000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += 0.02510539523571323;
                } else {
                  result[0] += -0.03158731111233469;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += -0.06046713836390491;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.07776618123665839;
                  } else {
                    result[0] += 0.015882414538811434;
                  }
                }
              }
            } else {
              result[0] += 0.009511258689737498;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
            result[0] += 0.007664835741377578;
          } else {
            result[0] += -0.07850725471072961;
          }
        }
      } else {
        result[0] += -0.04527598114537127;
      }
    } else {
      result[0] += -0.02006811579470809;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.043663788279399554;
          } else {
            result[0] += 0.006832306326637384;
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
            result[0] += 0.004566667992815435;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.86392068862915217) ) ) {
              result[0] += 0.021445085477939003;
            } else {
              result[0] += 0.09819166231836735;
            }
          }
        }
      } else {
        result[0] += -0.06793122730087676;
      }
    } else {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.824383735656740058) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.08770579383931;
          } else {
            result[0] += 0.005866619425161279;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
            result[0] += -0.029457173989399083;
          } else {
            result[0] += -0.07909034256844487;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.987184524536133701) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.05084543587175318;
              } else {
                result[0] += -0.10217936826628325;
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                result[0] += -0.04583524621923614;
              } else {
                result[0] += 0.04101817007619549;
              }
            }
          } else {
            result[0] += 0.002661407919490633;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.04345695247487216;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.022795601786015862;
                  } else {
                    result[0] += -0.02875813261968323;
                  }
                } else {
                  result[0] += -0.035235407976202594;
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.05612673682481328;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                  result[0] += -0.07029453078847177;
                } else {
                  result[0] += 0.04705267344340708;
                }
              }
            }
          } else {
            result[0] += -0.07046634250526539;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.352615833282471591) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                        result[0] += 0.013266227177845608;
                      } else {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.05095645586256623;
                        } else {
                          result[0] += -0.0989693205055423;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.02417852778509158;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.359572410583496982) ) ) {
                            result[0] += -0.10077995186356506;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                              result[0] += -0.10536208104185654;
                            } else {
                              result[0] += 0.02207324614514916;
                            }
                          }
                        } else {
                          result[0] += -0.18000498213973892;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.535362005233765537) ) ) {
                      result[0] += -0.08771302083860799;
                    } else {
                      result[0] += 0.030545832762958885;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
                    result[0] += 0.03996621341303473;
                  } else {
                    result[0] += 0.006890782647311454;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.556798219680787021) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                      result[0] += -0.0277114346086303;
                    } else {
                      result[0] += 0.07983245844474528;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                      result[0] += 0.03467000354859739;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.06602795223522702;
                        } else {
                          result[0] += 0.07091880668565063;
                        }
                      } else {
                        result[0] += -0.09540831734369475;
                      }
                    }
                  }
                } else {
                  result[0] += -0.04797831661714694;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)64.50000000000001421) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.901921629905701128) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
                      result[0] += -0.034905057608577365;
                    } else {
                      result[0] += 0.040037488698074386;
                    }
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
                      result[0] += 0.021755761828714474;
                    } else {
                      result[0] += -0.01886007184325939;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.06882500648498624) ) ) {
                    result[0] += 0.01111004492778944;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.03278044895550158;
                    } else {
                      result[0] += -0.08663567347841888;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                    result[0] += -0.04041580167125968;
                  } else {
                    result[0] += -0.0910202198478185;
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                    result[0] += 0.04028654488102837;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                      result[0] += 0.06077955011415224;
                    } else {
                      if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.062188261536753414;
                      } else {
                        result[0] += 0.029813287141740804;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.04098171109407707;
                } else {
                  result[0] += 0.01539217229630504;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                      result[0] += -0.08780557223724934;
                    } else {
                      result[0] += 0.04170896367096416;
                    }
                  } else {
                    result[0] += 0.03227814185042451;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.07249445109621931;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.06205601242788161;
                    } else {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.08422653377049182;
                      } else {
                        result[0] += 0.014879419369072694;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                  result[0] += -0.008170310840570296;
                } else {
                  result[0] += 0.028212395442306318;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.005064487833224097;
                  } else {
                    result[0] += 0.0241015952644727;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.012168035185488237;
                  } else {
                    result[0] += -0.07855840215288966;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
            result[0] += 0.007496075305025451;
          } else {
            result[0] += -0.07679744189162013;
          }
        }
      } else {
        result[0] += -0.03989182615829017;
      }
    } else {
      result[0] += -0.019427812532624315;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
          result[0] += -0.02977285808774456;
        } else {
          result[0] += -0.07419486976174693;
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.09192949987416406;
        } else {
          result[0] += -0.012203890967896253;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.400641441345215288) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.08071585996230829;
            } else {
              result[0] += -0.09288785678243516;
            }
          } else {
            result[0] += -0.05173013123537567;
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
            result[0] += 0.06515855294509282;
          } else {
            result[0] += 0.007498944887146693;
          }
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.06602573530623984;
          } else {
            result[0] += 0.004789344312393777;
          }
        } else {
          result[0] += -0.10337074899415546;
        }
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
          if ( UNLIKELY(  (data[63].missing != -1) && (data[63].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.1705754113921768;
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.06320177733702144;
                } else {
                  result[0] += -0.02352083824580866;
                }
              } else {
                result[0] += -0.06910569510374681;
              }
            } else {
              result[0] += -0.0014473543902937805;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.09131411944380839;
          } else {
            result[0] += 0.015278535203416802;
          }
        }
      } else {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)208.5000000000000284) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                      result[0] += -0.008843520116794025;
                    } else {
                      result[0] += 0.029362477845775267;
                    }
                  } else {
                    result[0] += -0.020698898528911644;
                  }
                } else {
                  result[0] += -0.03114518058894035;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.027910773537255296;
                } else {
                  result[0] += -0.005282856056257263;
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                  result[0] += 0.041985655444115005;
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.02002176674579266;
                  } else {
                    result[0] += -0.031890349032952146;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                  result[0] += -0.02512876832856197;
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.265274047851563388) ) ) {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.06374094068323492;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                        result[0] += -0.0876285255291785;
                      } else {
                        result[0] += 0.07047468006085007;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.018867349920271906;
                    } else {
                      result[0] += 0.07866741308922885;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05835151672363459) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                        result[0] += -0.07573436608649103;
                      } else {
                        result[0] += -0.0071305428957819696;
                      }
                    } else {
                      result[0] += 0.02274753021266486;
                    }
                  } else {
                    result[0] += 0.04038023046753655;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.716979026794434482) ) ) {
                    result[0] += 0.06501614649644344;
                  } else {
                    result[0] += -0.09278564005207324;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.07002640141208683;
                } else {
                  result[0] += -0.02134740944218996;
                }
              }
            } else {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)114.5000000000000142) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.623641014099121982) ) ) {
                        result[0] += -0.07068522595167563;
                      } else {
                        result[0] += 0.05365975391043052;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.549068689346314365) ) ) {
                        result[0] += 0.011161455488506021;
                      } else {
                        result[0] += -0.07001310258877959;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
                        result[0] += 0.02600206976362552;
                      } else {
                        result[0] += 0.08670117442134566;
                      }
                    } else {
                      result[0] += -0.005371728963416119;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.131513118743898261) ) ) {
                    result[0] += 0.02998304833443551;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                      if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.02952619786869172;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.01014741098813197;
                        } else {
                          result[0] += 0.07201827961027375;
                        }
                      }
                    } else {
                      result[0] += -0.040848714616640164;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                    result[0] += -0.035136668787446514;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        result[0] += -0.008630045261560948;
                      } else {
                        result[0] += 0.03832303998549823;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.795884609222413886) ) ) {
                        result[0] += 0.011958880529140415;
                      } else {
                        result[0] += -0.05018268503927001;
                      }
                    }
                  }
                } else {
                  result[0] += 0.016030483354429338;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.07136787628359136;
              } else {
                result[0] += 0.01316788007077187;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.027639389593548433;
                  } else {
                    result[0] += 0.06221798160551781;
                  }
                } else {
                  result[0] += -0.027923881516149097;
                }
              } else {
                result[0] += -0.06371895044360354;
              }
            }
          } else {
            result[0] += -0.07383994910288892;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.852773189544679511) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.030259191139907934;
        } else {
          result[0] += 0.032519712622742634;
        }
      } else {
        result[0] += -0.08346534333788287;
      }
    }
  } else {
    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += -0.006100332270816788;
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.009831496542864117;
        } else {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.09130681048058725;
          } else {
            result[0] += -0.037357176680278686;
          }
        }
      }
    } else {
      result[0] += -0.04950369701128802;
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)196.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.16383555706105624;
              } else {
                result[0] += 0.0008164914471193925;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.055548496757654275;
                } else {
                  result[0] += -0.028912718226674735;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                  result[0] += 0.02402750191889579;
                } else {
                  result[0] += -0.09322997394362456;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.189289569854737216) ) ) {
                result[0] += 0.060255962431304155;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                  result[0] += 0.08879755266625536;
                } else {
                  result[0] += -0.06999217169693277;
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.09441141416678556;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
                  result[0] += -0.06763592813432628;
                } else {
                  result[0] += 0.04173489232381134;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.011569780908060294;
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.012565418956141869;
              } else {
                result[0] += -0.05682456752336709;
              }
            }
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.029587522497575097;
            } else {
              result[0] += -0.08185475916462699;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
          result[0] += -0.012641786641965878;
        } else {
          result[0] += -0.0930493788854723;
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.239300251007080966) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.005974504154291616;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
            result[0] += -0.009242381970155417;
          } else {
            result[0] += -0.04581847334584556;
          }
        }
      } else {
        result[0] += -0.04788176285741613;
      }
    }
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39772605895996271) ) ) {
            result[0] += -0.025310174729739027;
          } else {
            result[0] += 0.030360112013797194;
          }
        } else {
          result[0] += -0.07142289210246858;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.312552452087403232) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  result[0] += 0.0001991357612128378;
                } else {
                  result[0] += -0.035304453026860694;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)218.5000000000000284) ) ) {
                    result[0] += -0.010004248401834913;
                  } else {
                    result[0] += 0.02779252562546737;
                  }
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.03299439858463274;
                    } else {
                      result[0] += 0.03224805435304611;
                    }
                  } else {
                    result[0] += 0.06798167312704533;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.021644578664875434;
              } else {
                result[0] += -0.10132636156293727;
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.347096204757691318) ) ) {
                result[0] += 0.03817458108939712;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.018060301639207976;
                } else {
                  result[0] += -0.026768733693176373;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.02164961347284719;
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.03403710286015938;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.07789868115321891;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                      result[0] += -0.07630735066239143;
                    } else {
                      result[0] += 0.07001334776026893;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
              if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.036363364858451;
              } else {
                result[0] += -0.011620262680765867;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.924581527709961826) ) ) {
                    result[0] += -0.061293169037091956;
                  } else {
                    result[0] += 0.013682705050775835;
                  }
                } else {
                  result[0] += -0.10803949480264059;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.07087352742742395;
                    } else {
                      result[0] += 0.0054527512690913426;
                    }
                  } else {
                    result[0] += 0.048148931318568504;
                  }
                } else {
                  result[0] += -0.0290555866531161;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += 0.01716764252868973;
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
                      result[0] += -0.02229051984774172;
                    } else {
                      result[0] += 0.02302728381648893;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.280352115631104404) ) ) {
                        result[0] += 0.009700074764985902;
                      } else {
                        result[0] += -0.06532245612856574;
                      }
                    } else {
                      result[0] += -0.06374178473010614;
                    }
                  }
                } else {
                  result[0] += 0.01419242734506884;
                }
              } else {
                result[0] += 0.015541742447971477;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
          result[0] += -0.006847794250798209;
        } else {
          result[0] += -0.08204980667531231;
        }
      } else {
        result[0] += 0.02279787356621164;
      }
    }
  }
  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += 0.15949478918048587;
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.00087734739390411;
            } else {
              result[0] += -0.03505185810594229;
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
              result[0] += 0.10507987620186308;
            } else {
              result[0] += -0.0788503503126274;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
              result[0] += 0.041841648212118526;
            } else {
              result[0] += -0.047965400323168525;
            }
          }
        }
      } else {
        result[0] += -0.05650099584982761;
      }
    } else {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.0005855390639971206;
          } else {
            result[0] += 0.08075628223287971;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.010954399744588547;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                    result[0] += -0.004333912266716001;
                  } else {
                    result[0] += -0.06730183853955636;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += -0.015688377986330992;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += 0.08532006916141596;
                  } else {
                    result[0] += -0.005097071886161056;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.09153780064037316;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                  result[0] += -0.027906517368839423;
                } else {
                  result[0] += 0.08919063717006272;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.02038608781570528;
            } else {
              result[0] += -0.06471322593417736;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.655387401580811435) ) ) {
            result[0] += -0.016050406902920485;
          } else {
            if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += -0.09034012698084776;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                result[0] += -0.09328595169782442;
              } else {
                result[0] += -0.018136168291457457;
              }
            }
          }
        } else {
          result[0] += -0.0697466394824723;
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
          result[0] += -0.007759785406770105;
        } else {
          result[0] += 0.07820944021650883;
        }
      } else {
        result[0] += -0.029970561361240735;
      }
    } else {
      if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.82380008697509943) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += -0.05830922264290374;
                } else {
                  result[0] += 0.0130912928787334;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.758822202682496005) ) ) {
                  result[0] += -0.006432449711038965;
                } else {
                  result[0] += -0.07360369704430764;
                }
              }
            } else {
              result[0] += -0.024579073116442748;
            }
          } else {
            result[0] += 0.0324488914394573;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.473832368850708896) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.363078355789185458) ) ) {
                  result[0] += -0.0030752190862484933;
                } else {
                  result[0] += 0.02437224964094962;
                }
              } else {
                result[0] += -0.03152888360806662;
              }
            } else {
              result[0] += 0.03730106708867636;
            }
          } else {
            if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.013228019803737416;
              } else {
                result[0] += -0.06979247149425065;
              }
            } else {
              if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.016224541946961667;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.02394164219824751;
                } else {
                  result[0] += 0.08093832028986157;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                result[0] += -0.01277107555205245;
              } else {
                result[0] += -0.05056293387514345;
              }
            } else {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                  if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.012453445539048617;
                    } else {
                      result[0] += -0.05060076038291346;
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.020127415657043901) ) ) {
                      result[0] += 0.0011749811516574924;
                    } else {
                      result[0] += -0.07604206975285883;
                    }
                  }
                } else {
                  result[0] += 0.020684835475625885;
                }
              } else {
                result[0] += -0.07093236968997034;
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                result[0] += 0.02773086598174454;
              } else {
                result[0] += -0.07143210246906295;
              }
            } else {
              result[0] += 0.01852658100486037;
            }
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.972562313079834873) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                result[0] += -0.06746227739117087;
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.014063855620938813;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += -0.006093195255852862;
                  } else {
                    result[0] += -0.09835082706848937;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)4.500000000000000888) ) ) {
                result[0] += 0.03228779431260376;
              } else {
                result[0] += -0.08810783638429037;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
              result[0] += -0.06613672306762244;
            } else {
              result[0] += 0.009822034912914133;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
                    result[0] += -0.005827243310199455;
                  } else {
                    result[0] += 0.03722603853933904;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)176.5000000000000284) ) ) {
                    result[0] += -0.05843626801697166;
                  } else {
                    result[0] += 0.011807235911532099;
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.0027920442996130996;
                    } else {
                      result[0] += -0.09003183236325285;
                    }
                  } else {
                    result[0] += 0.0009672753154805685;
                  }
                } else {
                  result[0] += 0.011989372244425235;
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.359572410583496982) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.08490453562373124;
                    } else {
                      result[0] += -0.017732625828507244;
                    }
                  } else {
                    result[0] += 0.06118174754863114;
                  }
                } else {
                  result[0] += 0.009984097149968311;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                    result[0] += 0.04958638693075529;
                  } else {
                    result[0] += 0.020841023274284606;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.607985973358155185) ) ) {
                    result[0] += 0.03211064006620197;
                  } else {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
                      result[0] += 0.007040716781579837;
                    } else {
                      result[0] += -0.04733193657033296;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)66.50000000000001421) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                    result[0] += -0.00011499656213139057;
                  } else {
                    result[0] += 0.03628521025473515;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += -0.10323214759823868;
                    } else {
                      result[0] += -0.00023842504772687416;
                    }
                  } else {
                    if ( UNLIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.12205905271477054;
                      } else {
                        result[0] += 0.005813982143937554;
                      }
                    } else {
                      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.04105095252684102;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.010457966038042608;
                        } else {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += -0.07774436762404952;
                          } else {
                            result[0] += 0.05522043553661356;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.06882500648498624) ) ) {
                  result[0] += -0.0013776677363504406;
                } else {
                  result[0] += -0.05828799682295879;
                }
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.08882532042660328;
              } else {
                result[0] += -0.02781461495722168;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)114.5000000000000142) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
              result[0] += 0.02155392568226332;
            } else {
              result[0] += 0.062306246229199995;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
              result[0] += 0.021458181008039215;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.74356317520141779) ) ) {
                result[0] += 0.008628293775517677;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.04593150949865514;
                } else {
                  result[0] += 0.011347541249462734;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12031126022339045) ) ) {
              result[0] += 0.0696104484924624;
            } else {
              result[0] += -0.03671124191150729;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51918649673462092) ) ) {
              result[0] += -0.03999563954266319;
            } else {
              result[0] += -0.09935851059676899;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
            result[0] += -0.016436356697686654;
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.009348536072385788;
              } else {
                result[0] += -0.09751110673268804;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0638248599799459;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.04170410120449082;
                } else {
                  result[0] += 0.04009178465697957;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.017841336260036682;
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
          result[0] += -0.024276036589833162;
        } else {
          result[0] += -0.068381948119211;
        }
      } else {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)1.00000001800250948e-35) ) ) {
          result[0] += -0.08894147350303142;
        } else {
          result[0] += -0.01154158921533917;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)2.500000000000000444) ) ) {
          result[0] += 0.05773613533554448;
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += 0.04069481080842497;
            } else {
              result[0] += -0.04815509733075513;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.037429118768321716;
              } else {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += 0.038677574426249646;
                  } else {
                    result[0] += -0.06213055617497233;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.0656546676036852;
                  } else {
                    result[0] += 0.027658843749227453;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += -0.006916036778125439;
              } else {
                result[0] += -0.07803430931691407;
              }
            }
          }
        }
      } else {
        result[0] += -0.06989237256957358;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.461708784103394443) ) ) {
            result[0] += -0.008030355835258352;
          } else {
            result[0] += -0.04737247822089469;
          }
        } else {
          result[0] += -0.004063463913261878;
        }
      } else {
        if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
          result[0] += 0.00845274909790892;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.132412433624269354) ) ) {
            if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.0837453335839094;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.02379255135879333;
              } else {
                result[0] += 0.013318651629745035;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.388237953186036044) ) ) {
                if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.05249750553549191;
                } else {
                  result[0] += 0.022315564972353867;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95053911209106623) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
                    result[0] += 0.0035465389341405697;
                  } else {
                    result[0] += 0.039487271260551814;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.07857085032257499;
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.04279241309108966;
                    } else {
                      result[0] += 0.06176568947174204;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.0785023402275163;
              } else {
                result[0] += 0.008694043973432706;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.0657889371747898;
              } else {
                result[0] += 0.04534769179064761;
              }
            } else {
              result[0] += 0.0467182324123491;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.01359977774897439;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                      result[0] += 0.06771786180211921;
                    } else {
                      result[0] += -0.0005342497714278312;
                    }
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.07862488833027471;
                      } else {
                        result[0] += -0.026863229351694485;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.67574596405029475) ) ) {
                        result[0] += -0.049355581442863325;
                      } else {
                        result[0] += 0.10779630488931063;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.987184524536133701) ) ) {
                  result[0] += -0.046992697669135775;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                    result[0] += 0.04740251343792857;
                  } else {
                    result[0] += -0.032398419358272666;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.693829536437990058) ) ) {
                result[0] += 0.03316572812528474;
              } else {
                if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.023344932187753683;
                  } else {
                    result[0] += -0.070054170544283;
                  }
                } else {
                  if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                      result[0] += -0.07442228887787976;
                    } else {
                      result[0] += 0.011457760077291107;
                    }
                  } else {
                    result[0] += 0.03894423301338795;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.358708143234253818) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)214.5000000000000284) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.002865038054752955;
                    } else {
                      result[0] += -0.06564307634646456;
                    }
                  } else {
                    result[0] += 0.02262638893195513;
                  }
                } else {
                  result[0] += 0.03758541782815747;
                }
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.00668676221944783;
                } else {
                  result[0] += 0.0679188650529226;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.03049076645587877;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.011070026461464573;
                } else {
                  result[0] += -0.06477486036276318;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.767324447631837714) ) ) {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.358708143234253818) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                      result[0] += -0.01744362509212681;
                    } else {
                      result[0] += 0.10532402970440691;
                    }
                  } else {
                    result[0] += 0.02908538302782467;
                  }
                } else {
                  result[0] += -0.013980934990260483;
                }
              } else {
                result[0] += -0.009686666842881317;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
                  if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                        result[0] += -0.04846697439131506;
                      } else {
                        result[0] += 0.014159583662636657;
                      }
                    } else {
                      result[0] += 0.04738603103038383;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.053050994873047763) ) ) {
                      result[0] += 0.009537469826365445;
                    } else {
                      result[0] += -0.050467438500864416;
                    }
                  }
                } else {
                  result[0] += -0.03862663052649501;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                  result[0] += 0.027253408767733613;
                } else {
                  result[0] += -0.07357800081835603;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.852773189544679511) ) ) {
          result[0] += 0.00284749477918482;
        } else {
          result[0] += -0.0743832843329233;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += -0.002831855016420302;
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.008545592236938892;
        } else {
          result[0] += -0.047498797373470195;
        }
      }
    } else {
      result[0] += -0.04151342557451159;
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)196.5000000000000284) ) ) {
            if ( UNLIKELY(  (data[34].missing != -1) && (data[34].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += 0.15256173719849592;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.044774536505011286;
                } else {
                  result[0] += -0.03067887936423075;
                }
              } else {
                result[0] += 0.0001382269126294302;
              }
            }
          } else {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                  result[0] += 0.06660350743384028;
                } else {
                  result[0] += -0.032169634812477525;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                  result[0] += 0.049294882839413474;
                } else {
                  result[0] += -0.07502573869833316;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                result[0] += 0.015436254142782832;
              } else {
                result[0] += -0.0920516689829342;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.493027687072754794) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)120.5000000000000142) ) ) {
                  result[0] += -0.00017501938808416528;
                } else {
                  result[0] += 0.026463640011734037;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
                        result[0] += -0.007392838033387814;
                      } else {
                        result[0] += 0.023376687108367267;
                      }
                    } else {
                      result[0] += -0.0401318640861679;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
                      result[0] += 0.02275225875287717;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                        result[0] += 0.018011220190368698;
                      } else {
                        result[0] += -0.04404544101559515;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)124.5000000000000142) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.010110576897829779;
                    } else {
                      result[0] += -0.09237257954721005;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.006537906344223097;
                      } else {
                        result[0] += -0.04886082661087295;
                      }
                    } else {
                      result[0] += -0.05649110410973183;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.005576846551739855;
                    } else {
                      result[0] += -0.0637559065049541;
                    }
                  } else {
                    result[0] += -0.0743684675672088;
                  }
                } else {
                  result[0] += 0.017481601394406958;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.02379002960887496;
                  } else {
                    result[0] += 0.037240309860194824;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.06723501438220157;
                  } else {
                    result[0] += 0.020835468989927913;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += -0.05704468006736524;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                result[0] += -0.025081308987088005;
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.051047984625715276;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.041401422588373094;
                  } else {
                    result[0] += -0.041419988147020984;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.558241367340089667) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.01769001965813933;
          } else {
            result[0] += 0.03602440546267094;
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.09612999094477777;
          } else {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.02318298109731328;
            } else {
              result[0] += 0.8649177706889395;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.645740747451783115) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.000307083129883701) ) ) {
                result[0] += -0.021328229042600004;
              } else {
                result[0] += -0.10377838795668977;
              }
            } else {
              result[0] += 0.020298898930046487;
            }
          } else {
            result[0] += -0.007198979624316933;
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.044975880606671244;
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.04787783812435104;
                } else {
                  result[0] += -0.03333334913234672;
                }
              } else {
                result[0] += 0.07458617868266983;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.0871358912351351;
            } else {
              result[0] += -0.02050325949086372;
            }
          }
        }
      } else {
        result[0] += -0.06849214279556189;
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
          result[0] += 0.033744172242461394;
        } else {
          result[0] += -0.002692566079087962;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.0024925660083288457;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.17027091979980646) ) ) {
                result[0] += -0.03716637617840022;
              } else {
                result[0] += -0.12686102431773258;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.28931427001953303) ) ) {
                result[0] += -0.023438985366755295;
              } else {
                result[0] += 0.11504128076135044;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
            result[0] += -0.0767789515354948;
          } else {
            result[0] += -0.030723003352083802;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
        result[0] += -0.07936238314872048;
      } else {
        result[0] += -0.03122662799883101;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)5.500000000000000888) ) ) {
        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)51.50000000000000711) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.684390544891359198) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.280352115631104404) ) ) {
                    result[0] += -0.07663519517023841;
                  } else {
                    result[0] += 0.017727414141197658;
                  }
                } else {
                  result[0] += -0.000326148316723577;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.02230695024087727;
                  } else {
                    result[0] += -0.06929669531533089;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95053911209106623) ) ) {
                      result[0] += 0.014380946787566566;
                    } else {
                      result[0] += 0.06779380946822272;
                    }
                  } else {
                    result[0] += -0.03687226026449957;
                  }
                }
              }
            } else {
              result[0] += 0.00937531218804567;
            }
          } else {
            if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                result[0] += -0.0016874450254668997;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.924915313720704901) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.06851572012572608;
                  } else {
                    result[0] += 0.005295383995066297;
                  }
                } else {
                  if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                    result[0] += -0.03791486171194512;
                  } else {
                    result[0] += -0.10199307202908983;
                  }
                }
              }
            } else {
              result[0] += 0.0008973013954384313;
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
            result[0] += 0.0021854130802035593;
          } else {
            result[0] += -0.019835516130388725;
          }
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += -0.09366678329305789;
        } else {
          result[0] += 0.00045297433207751314;
        }
      }
    } else {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)53.50000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.05894855393336743;
              } else {
                result[0] += 0.013835817701908571;
              }
            } else {
              result[0] += 0.018021439923529527;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
              result[0] += -0.050102356093325806;
            } else {
              result[0] += 0.017793701765153363;
            }
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.011061547977597515;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      result[0] += 0.015308868816803822;
                    } else {
                      result[0] += -0.04519649427656624;
                    }
                  }
                } else {
                  result[0] += -0.043565178921324776;
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.987184524536133701) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.689592361450196201) ) ) {
                    result[0] += 0.03718003355799661;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.030025957337562754;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.010496067285109992;
                      } else {
                        result[0] += -0.06266166669006358;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
                      result[0] += -0.005734662916950034;
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                        result[0] += 0.02576253406735989;
                      } else {
                        result[0] += -0.07597425639247336;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.767332553863526279) ) ) {
                      result[0] += 0.06672504725403561;
                    } else {
                      result[0] += -0.03768102756152206;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                result[0] += 0.02116309404059048;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.006063950405154385;
                  } else {
                    if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.08829904210046133;
                    } else {
                      result[0] += -0.02211414875596494;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.08340214004292787;
                  } else {
                    result[0] += -0.03457033909470881;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.607985973358155185) ) ) {
                    result[0] += -0.04416830176422588;
                  } else {
                    result[0] += 0.018406077404048222;
                  }
                } else {
                  result[0] += -0.07332744611215825;
                }
              } else {
                result[0] += 0.015252680511710016;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.010199241388336049;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.825982809066773349) ) ) {
                    result[0] += 0.017316471015960785;
                  } else {
                    result[0] += 0.06349340874243363;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  result[0] += -0.07052826962238605;
                } else {
                  result[0] += 0.013878107012001026;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.57868480682373225) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += 0.05167177288097374;
            } else {
              result[0] += -0.024596743246486343;
            }
          } else {
            result[0] += -0.06848406383697114;
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.74356317520141779) ) ) {
            result[0] += -0.008843773816470557;
          } else {
            if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.061142121620371775;
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.06398311427966895;
              } else {
                result[0] += 0.0005709034066589583;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
        result[0] += -0.0025811486384847144;
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.008155868995632866;
        } else {
          result[0] += -0.04400935615939645;
        }
      }
    } else {
      result[0] += -0.038121735919643324;
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.467161655426027167) ) ) {
              result[0] += 0.0037022929207358463;
            } else {
              result[0] += -0.05638805519885061;
            }
          } else {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)115.5000000000000142) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.04908582530926693;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.10792179043939239;
                  } else {
                    result[0] += 0.024039123838894623;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.07176939569993009;
                } else {
                  if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                      if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.08360127362444714;
                      } else {
                        result[0] += 0.01988229219020704;
                      }
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.03132245164047892;
                      } else {
                        result[0] += -0.053333020020651836;
                      }
                    }
                  } else {
                    result[0] += 0.034334241055720124;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.011641126752035314;
                  } else {
                    result[0] += -0.023474913113440892;
                  }
                } else {
                  result[0] += 0.03092997762825609;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                    result[0] += -0.00603660745040481;
                  } else {
                    result[0] += -0.0771973866865561;
                  }
                } else {
                  result[0] += 0.004378674390735116;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.0835146903991717) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)110.5000000000000142) ) ) {
                result[0] += -0.02249791521459076;
              } else {
                result[0] += 0.010415293389773754;
              }
            } else {
              if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.037358908528585186;
              } else {
                result[0] += -0.01596868815591562;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                result[0] += 0.009705219970763438;
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.009285309189078812;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.549646615982056552) ) ) {
                    result[0] += 0.020091765451307055;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.07008973589964797;
                    } else {
                      result[0] += -0.016011376336917574;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += -0.04917275006191235;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += 0.05093247152782648;
                } else {
                  result[0] += -0.022356166250379585;
                }
              }
            }
          }
        }
      } else {
        result[0] += -0.03320587720471679;
      }
    } else {
      result[0] += -0.014750918982903836;
    }
  } else {
    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.03521020460508167;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += 0.013807057745558888;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
                  result[0] += -0.04295543036108506;
                } else {
                  result[0] += 0.00591358402648191;
                }
              }
            } else {
              result[0] += 0.04946855904207749;
            }
          }
        } else {
          if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)12.00000000000000178) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              result[0] += 0.03433649055097588;
            } else {
              result[0] += -0.06073040694961668;
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.03507544100246093;
            } else {
              result[0] += 0.04292357021231456;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
          result[0] += -0.06157835504698265;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
            result[0] += -0.06465392832689024;
          } else {
            result[0] += 0.11069946099200012;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.07286570822209086;
          } else {
            result[0] += 0.007378024130263006;
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.022793136334080505;
            } else {
              result[0] += -0.0713919185890116;
            }
          } else {
            result[0] += -0.0825141381729868;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.313104629516603339) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += -0.025285489004913644;
              } else {
                result[0] += -0.062187723229047864;
              }
            } else {
              result[0] += -0.106516727345866;
            }
          } else {
            result[0] += 0.014342443566382729;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.08672893709476544;
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.0017592157597685243;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
                      result[0] += -0.024018370314704985;
                    } else {
                      result[0] += 0.048635325056293276;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.07809198377232707;
                    } else {
                      result[0] += 0.0061132694492555244;
                    }
                  } else {
                    result[0] += 0.007993176485221137;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.02798794389994755;
              } else {
                result[0] += 0.10258975735131821;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.08167837124263;
            } else {
              result[0] += -0.002315405064690013;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)196.5000000000000284) ) ) {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)174.5000000000000284) ) ) {
                if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.1521079665625319;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.23832273483276456) ) ) {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                        if ( LIKELY( !(data[46].missing != -1) || (data[46].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.07612928606696535;
                        } else {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)31.50000000000000355) ) ) {
                            result[0] += -0.06692759833537348;
                          } else {
                            result[0] += 0.053061776651641225;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.03315177931723679;
                        } else {
                          result[0] += 0.052347911125461;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)70.50000000000001421) ) ) {
                          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += 0.059486201445184086;
                          } else {
                            result[0] += 0.12847747073613763;
                          }
                        } else {
                          result[0] += 0.23657238033179687;
                        }
                      } else {
                        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                          result[0] += -0.022221220355849287;
                        } else {
                          result[0] += 0.05204659554583875;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
                        result[0] += 0.005036599635252618;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.543220520019532138) ) ) {
                          if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.006731538338516548;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.50511837005615412) ) ) {
                              result[0] += 0.06694835609111088;
                            } else {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.90263271331787287) ) ) {
                                result[0] += 0.013246623181757247;
                              } else {
                                result[0] += -0.055123079286848134;
                              }
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                            result[0] += 0.019494338391338278;
                          } else {
                            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.067782521247864214) ) ) {
                                result[0] += -0.019098027342168405;
                              } else {
                                result[0] += -0.08073821129095275;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.854362010955811435) ) ) {
                                result[0] += -0.0443749866390657;
                              } else {
                                result[0] += -0.0981968564237723;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.386624813079835761) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                            if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)5.500000000000000888) ) ) {
                              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                                result[0] += 0.040691608006206914;
                              } else {
                                result[0] += -0.029549778170367635;
                              }
                            } else {
                              result[0] += -0.09494478991503803;
                            }
                          } else {
                            result[0] += 0.04559178944384656;
                          }
                        } else {
                          result[0] += -0.08546117570493622;
                        }
                      } else {
                        result[0] += -0.07546157332812328;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
                  result[0] += 0.10365194601142436;
                } else {
                  result[0] += 0.024227142724596022;
                }
              }
            } else {
              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                    result[0] += 0.0575475198851824;
                  } else {
                    result[0] += -0.03537355837046968;
                  }
                } else {
                  result[0] += -0.027866159723406454;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.635775566101075995) ) ) {
                  result[0] += 0.001802257190171261;
                } else {
                  result[0] += -0.09513376864200569;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += -0.002137176546085896;
              } else {
                result[0] += 0.017386469179130237;
              }
            } else {
              if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.011098269252425519;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                  result[0] += 0.00794845000230941;
                } else {
                  result[0] += -0.07026511148776632;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)188.5000000000000284) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                result[0] += 0.005092064929289522;
              } else {
                result[0] += -0.05844433745990794;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.77165889739990412) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)15.50000000000000178) ) ) {
                  result[0] += -0.08359756025756829;
                } else {
                  result[0] += -0.010049112449367554;
                }
              } else {
                if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                      result[0] += 0.08989028433141877;
                    } else {
                      result[0] += -0.007824225096909266;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.029730766355371294;
                    } else {
                      result[0] += 0.08521209837844536;
                    }
                  }
                } else {
                  result[0] += -0.05551831909131584;
                }
              }
            }
          } else {
            result[0] += -0.06336785729042303;
          }
        }
      } else {
        result[0] += -0.06914378234216921;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.766185760498047763) ) ) {
            result[0] += 0.005411717289313507;
          } else {
            result[0] += -0.05000607921327756;
          }
        } else {
          result[0] += 0.02712919264260117;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.802901029586792436) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.09619131325715759;
          } else {
            result[0] += -0.02399962141850899;
          }
        } else {
          result[0] += 0.17498290804984454;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
          result[0] += 0.03258419848793367;
        } else {
          result[0] += -0.0025260100508742844;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.0032234782014872554;
        } else {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.08276129902866021;
          } else {
            result[0] += -0.03182557345969408;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
        result[0] += -0.07517324803977507;
      } else {
        result[0] += -0.0315905370334325;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)96.00000000000001421) ) ) {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY(  (data[31].missing != -1) && (data[31].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.14986906486843318;
        } else {
          if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.726826429367066318) ) ) {
              if ( LIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.828906774520874912) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.297262430191040927) ) ) {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.586156606674195224) ) ) {
                          result[0] += 0.021870917350662274;
                        } else {
                          if ( LIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += -0.04507700991624308;
                          } else {
                            result[0] += 0.017874678528427876;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                          result[0] += 0.02925951627056063;
                        } else {
                          if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.08469193418672584;
                          } else {
                            result[0] += -0.035316886358344665;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.419101238250734198) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.303533792495728427) ) ) {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)70.50000000000001421) ) ) {
                            result[0] += 0.05647015231802748;
                          } else {
                            result[0] += 0.13459460272471235;
                          }
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[53].missing != -1) || (data[53].fvalue <= (double)2.500000000000000444) ) ) {
                              result[0] += 0.019181509271978633;
                            } else {
                              result[0] += -0.0312336046468325;
                            }
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                              if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                                result[0] += 0.02557099253799354;
                              } else {
                                result[0] += 0.09705791380992482;
                              }
                            } else {
                              result[0] += 0.020618112268609465;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                          result[0] += 0.014592920699040626;
                        } else {
                          result[0] += -0.05950694263192753;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.09922462910968963;
                      } else {
                        result[0] += -0.05399090552973077;
                      }
                    } else {
                      result[0] += -0.024675929668548553;
                    }
                  }
                } else {
                  result[0] += -0.05098543094090302;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)92.50000000000001421) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.197173833847046787) ) ) {
                        result[0] += -0.022151596825632263;
                      } else {
                        result[0] += 0.011969747600752086;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                        result[0] += 0.08436104719275431;
                      } else {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.06855689664890556;
                        } else {
                          result[0] += -0.03045679472379205;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.06706513890746761;
                  }
                } else {
                  result[0] += 0.006240617861804883;
                }
              }
            } else {
              if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                    if ( UNLIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.03761733323631593;
                    } else {
                      result[0] += 0.0076194677908679835;
                    }
                  } else {
                    if ( UNLIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                        result[0] += 0.03984459409905105;
                      } else {
                        result[0] += -0.0928642597736074;
                      }
                    } else {
                      result[0] += -0.0813328861624687;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                    if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)2.500000000000000444) ) ) {
                      result[0] += -0.04813128231228903;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.531409263610840732) ) ) {
                        result[0] += 0.06463420197559601;
                      } else {
                        result[0] += -0.026115578347223636;
                      }
                    }
                  } else {
                    result[0] += -0.09282288890870177;
                  }
                }
              } else {
                if ( LIKELY( !(data[64].missing != -1) || (data[64].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.436733961105347568) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.597130775451661044) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += -0.0839596988605532;
                        } else {
                          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.049352141142132294;
                          } else {
                            result[0] += -0.05567426409347414;
                          }
                        }
                      } else {
                        result[0] += 0.04140339309948651;
                      }
                    } else {
                      result[0] += 0.07161114402375343;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                      result[0] += -0.08089589061730411;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.059420347213746005) ) ) {
                        result[0] += 0.019603452320871254;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.06239572473991552;
                        } else {
                          result[0] += 0.044875217557247854;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.06842540886112257;
                  } else {
                    result[0] += 0.008393151378910465;
                  }
                }
              }
            }
          } else {
            result[0] += 0.005965570562247214;
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.940192461013794833) ) ) {
              result[0] += -0.03427366634037455;
            } else {
              result[0] += 0.03263746314052867;
            }
          } else {
            result[0] += 0.03164273443208029;
          }
        } else {
          result[0] += -0.06648267955613957;
        }
      }
    } else {
      result[0] += -0.013177440890939638;
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
          result[0] += 0.029264385067647003;
        } else {
          result[0] += -0.002866680783744744;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0010769495514705513;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              result[0] += -0.06727158447479069;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                result[0] += -0.03973557326197788;
              } else {
                result[0] += 0.06570628257694626;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
            result[0] += -0.0814821907557542;
          } else {
            result[0] += -0.032043910971211;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
        result[0] += -0.07311597031973739;
      } else {
        result[0] += -0.029912377669107787;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          result[0] += 0.14798755092902346;
        } else {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)21.50000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.75531578063965021) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.0806691697583314;
                } else {
                  result[0] += 0.005046056337686702;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.56941866874694913) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.434600353240968573) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.07536356363110083;
                      } else {
                        result[0] += 0.020153865317607263;
                      }
                    } else {
                      result[0] += -0.0018085773382039542;
                    }
                  } else {
                    result[0] += 0.023969881360912066;
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    result[0] += -0.06816680288007035;
                  } else {
                    result[0] += 0.0076855544120222295;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.06301373583097855;
              } else {
                if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.08341596029205874;
                } else {
                  result[0] += 0.008806853181916364;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.53813362121582209) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.742733001708986151) ) ) {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)54.50000000000000711) ) ) {
                            result[0] += -0.033695708758654346;
                          } else {
                            if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.01749413847774565;
                            } else {
                              result[0] += -0.009715894005621804;
                            }
                          }
                        } else {
                          result[0] += 0.015187706074498084;
                        }
                      } else {
                        if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
                            result[0] += 0.019182231871609574;
                          } else {
                            result[0] += -0.03349246966936037;
                          }
                        } else {
                          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                            result[0] += 0.05582469408362835;
                          } else {
                            result[0] += 0.024143631008435272;
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                          result[0] += -0.01607784346887108;
                        } else {
                          result[0] += -0.06247359991613602;
                        }
                      } else {
                        if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.0471045390662192;
                        } else {
                          result[0] += 0.023431147831571065;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.018754886573102418;
                      } else {
                        result[0] += -0.06489798776689833;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                        result[0] += -0.07405883400473269;
                      } else {
                        if ( LIKELY( !(data[62].missing != -1) || (data[62].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.05933821920985893;
                        } else {
                          result[0] += -0.007825793137629856;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.016833924775013286;
                      } else {
                        result[0] += 0.03280557233606722;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                        result[0] += 0.016629352686928136;
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.0009175058429744792;
                        } else {
                          result[0] += -0.07493807813180019;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                          result[0] += -0.10736357774624378;
                        } else {
                          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
                            result[0] += 0.0035629678594561232;
                          } else {
                            result[0] += -0.0670945362898952;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                          result[0] += 0.006197938532172607;
                        } else {
                          result[0] += -0.07875365133881561;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                            result[0] += 0.022581088632122923;
                          } else {
                            result[0] += -0.092430464875153;
                          }
                        } else {
                          result[0] += 0.011239654501688123;
                        }
                      } else {
                        result[0] += 0.06444502990976554;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.029102325680992854;
                    } else {
                      result[0] += -0.030667614774359017;
                    }
                  } else {
                    result[0] += 0.04233112991381355;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)211.5000000000000284) ) ) {
                      result[0] += -0.013541026721333383;
                    } else {
                      result[0] += -0.06460742755316849;
                    }
                  } else {
                    result[0] += -0.07270344128580772;
                  }
                }
              }
            } else {
              result[0] += -0.0034692666162495934;
            }
          }
        }
      } else {
        result[0] += -0.06685720619302556;
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
          if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)113.5000000000000142) ) ) {
            result[0] += -0.01577726294952967;
          } else {
            result[0] += 0.03919738632804808;
          }
        } else {
          result[0] += -0.025403260622336218;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
          result[0] += -0.09725945908871941;
        } else {
          result[0] += -0.02014856685068423;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
          result[0] += 0.028140240439039295;
        } else {
          result[0] += -0.002476044505274908;
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
          result[0] += -0.006387536968022952;
        } else {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.09044624290366943;
          } else {
            result[0] += -0.033290739704902615;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)116.5000000000000142) ) ) {
        result[0] += -0.07107248527633908;
      } else {
        result[0] += -0.028387443534125385;
      }
    }
  }
  if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)6.000000000000000888) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
      if ( UNLIKELY(  (data[63].missing != -1) && (data[63].fvalue <= (double)-1.00000001800250948e-35) ) ) {
        result[0] += 0.14345290146716796;
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
              result[0] += -0.004704872950169715;
            } else {
              result[0] += -0.042219062289683994;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.007973771536463675;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.06083404028245048;
                } else {
                  result[0] += 0.013934531643808482;
                }
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.05933130425920441;
              } else {
                result[0] += 0.008190510704251924;
              }
            }
          }
        } else {
          result[0] += 0.0007305580124120898;
        }
      }
    } else {
      if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.026417016983033115) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
                result[0] += 0.04359827600976099;
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                    result[0] += 0.005962977220007704;
                  } else {
                    result[0] += -0.0252856891574569;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    result[0] += -0.02177298301689733;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                        result[0] += -0.013766788323013833;
                      } else {
                        result[0] += 0.03736365352804445;
                      }
                    } else {
                      result[0] += 0.038257802969560495;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += 0.05149839385309351;
                } else {
                  result[0] += -0.00028589603978371284;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.0701485575423825;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                    result[0] += -0.03715396852194093;
                  } else {
                    if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.07217285547097008;
                    } else {
                      result[0] += 0.05246734399151998;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.025120538524310572;
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.658699750900269443) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.689592361450196201) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.590987443923951083) ) ) {
                  result[0] += 0.07371683320278162;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)62.50000000000000711) ) ) {
                    result[0] += 0.05403296777238963;
                  } else {
                    result[0] += -0.004762069801299711;
                  }
                }
              } else {
                result[0] += 0.02265426814680853;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                result[0] += -0.02622664264215951;
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)129.5000000000000284) ) ) {
                  result[0] += 0.033485737945727234;
                } else {
                  result[0] += -0.007979838794793588;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.03807656065307412;
            } else {
              if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += -0.03037676749725725;
              } else {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.027148787409063013;
                } else {
                  result[0] += 0.05702557888862406;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
                  result[0] += 0.007683601814417418;
                } else {
                  result[0] += 0.033304574154390365;
                }
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.448887825012207919) ) ) {
                  result[0] += -0.03374998463563463;
                } else {
                  result[0] += 0.06152478153749872;
                }
              }
            } else {
              if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.010158968829908119;
              } else {
                result[0] += 0.06626950977193931;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.039886234623617976;
              } else {
                result[0] += -0.011501536657596734;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.02490391980091976;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.603942871093750888) ) ) {
                    result[0] += -0.008402567898505747;
                  } else {
                    result[0] += -0.0532338201736946;
                  }
                }
              } else {
                result[0] += -0.06987957005338828;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.653507709503174716) ) ) {
                if ( UNLIKELY( !(data[41].missing != -1) || (data[41].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.028097423681017017;
                } else {
                  result[0] += 0.006328913292292961;
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += 0.0012387671003280115;
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)23.50000000000000355) ) ) {
                    result[0] += 0.03636633895031737;
                  } else {
                    result[0] += -0.038498666001376296;
                  }
                }
              }
            } else {
              result[0] += -0.030701989906224494;
            }
          } else {
            if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)149.5000000000000284) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.02999341797656982;
                } else {
                  result[0] += -0.02954632670734991;
                }
              } else {
                result[0] += -0.0259970482693993;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.0063792752262160635;
              } else {
                result[0] += -0.05623341316170798;
              }
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.07255280579320372;
      } else {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.001306066841390179;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.119004011154175693) ) ) {
            result[0] += -0.007616401899288323;
          } else {
            result[0] += -0.039826883856068945;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
        result[0] += -0.08235252118492052;
      } else {
        result[0] += -0.027060137803031504;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[36].missing != -1) && (data[36].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.13439602806937564;
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)20.50000000000000355) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.241300821304322177) ) ) {
            result[0] += -0.048543784382160865;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95053911209106623) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.114358901977539951) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += -0.07508727169141385;
                } else {
                  result[0] += -0.0031649749304066134;
                }
              } else {
                result[0] += -0.002821379908556366;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.08937131274227053;
              } else {
                result[0] += -0.010303692566506612;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.319199085235596591) ) ) {
                  result[0] += 0.009158565287043438;
                } else {
                  result[0] += -0.01876158579342549;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.547126770019532138) ) ) {
                  if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.012931327777370289;
                  } else {
                    result[0] += -0.044005210284308253;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.205894470214845526) ) ) {
                    result[0] += -0.07164472955116076;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.0482335772277732;
                    } else {
                      result[0] += -0.002637238063305363;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.548691272735597479) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.610357046127320224) ) ) {
                  result[0] += 0.022066470204533385;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.0017977779151063288;
                  } else {
                    result[0] += -0.06979558232533643;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                      result[0] += -0.09898907256234594;
                    } else {
                      if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)149.5000000000000284) ) ) {
                        result[0] += 0.011486344815827929;
                      } else {
                        result[0] += -0.058037420448868095;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += 0.011258738860198361;
                    } else {
                      result[0] += -0.0722572111138197;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.002880402269663282;
                  } else {
                    result[0] += 0.06648793819981255;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.026453888828012636;
                  } else {
                    result[0] += -0.02956605211201744;
                  }
                } else {
                  result[0] += 0.038112161725743145;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                  result[0] += -0.04767751901518865;
                } else {
                  result[0] += 0.0020099297641106915;
                }
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.602003335952759233) ) ) {
                  result[0] += 0.0046439637858684745;
                } else {
                  result[0] += -0.07966014808473007;
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                  result[0] += 0.04524043794300485;
                } else {
                  result[0] += -0.03741091278701597;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
          if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                result[0] += 0.00567538042397811;
              } else {
                result[0] += -0.054795966394321195;
              }
            } else {
              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.080726039135935;
              } else {
                if ( UNLIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.029830410921581757;
                } else {
                  result[0] += -0.048045578284859565;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += -0.0025937437546070287;
            } else {
              result[0] += 0.050272213725862824;
            }
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.473832368850708896) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.61636352539062678) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                      result[0] += -0.02426658922345898;
                    } else {
                      result[0] += 0.03655966581159888;
                    }
                  } else {
                    result[0] += -0.09764732584583519;
                  }
                } else {
                  result[0] += 0.03305810255510674;
                }
              } else {
                result[0] += 0.05484203816480476;
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)95.50000000000001421) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.655387401580811435) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
                      result[0] += -0.07474855761843703;
                    } else {
                      result[0] += -0.0023220244890369062;
                    }
                  } else {
                    result[0] += 0.0281122195687463;
                  }
                } else {
                  result[0] += 0.06202125829652049;
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.07723365743525093;
                } else {
                  result[0] += -0.017038329103159902;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
              result[0] += -0.07943973017398259;
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.03919025509769787;
              } else {
                result[0] += -0.06529102474580843;
              }
            }
          }
        }
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
      result[0] += -0.004574160284187956;
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.968900680541993964) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
              result[0] += -0.09045853593751807;
            } else {
              result[0] += -0.00015082818210314364;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.512576580047609198) ) ) {
              result[0] += -0.00944830984647181;
            } else {
              result[0] += -0.10091534240891564;
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.04915441153958303;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.007774996301572596;
            } else {
              result[0] += 0.08936065787062264;
            }
          }
        }
      } else {
        result[0] += -0.059705595666331907;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)6.000000000000000888) ) ) {
      result[0] += 0.13057090014387038;
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[60].missing != -1) || (data[60].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.009974587345292255;
        } else {
          if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
            if ( LIKELY( !(data[49].missing != -1) || (data[49].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                    result[0] += 0.0029986716733273987;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.461708784103394443) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.08016401383356583;
                      } else {
                        result[0] += -0.0535985588860516;
                      }
                    } else {
                      result[0] += -0.08723178079542276;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.38689327239990412) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.060410704052995604;
                          } else {
                            result[0] += 0.005901159715908332;
                          }
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.08719797898441203;
                          } else {
                            result[0] += -0.015417505923761055;
                          }
                        }
                      } else {
                        result[0] += -0.03474883128923686;
                      }
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                          result[0] += 0.04884849343566585;
                        } else {
                          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.014852772087738608;
                          } else {
                            result[0] += 0.08496414985674729;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
                            result[0] += 0.0039152814765088546;
                          } else {
                            result[0] += -0.06936864794560997;
                          }
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.08835079555510401;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                              result[0] += -0.18308877716659444;
                            } else {
                              result[0] += 0.07109213148084377;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.89399480819702326) ) ) {
                      if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.607985973358155185) ) ) {
                          result[0] += 0.06413472680248417;
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += 0.04614767539901259;
                            } else {
                              result[0] += -0.1312899596386188;
                            }
                          } else {
                            result[0] += -0.007313963267419167;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
                          result[0] += 0.008698812316035583;
                        } else {
                          result[0] += 0.026085704713517927;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0201649770307027;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                          result[0] += -0.09432017505887821;
                        } else {
                          result[0] += 0.0719880075687681;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[56].missing != -1) || (data[56].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.004885954549184168;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                        result[0] += -0.006505163732865781;
                      } else {
                        result[0] += 0.03604829491867966;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.632002353668214667) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.013107634652368527;
                        } else {
                          result[0] += -0.015609750533745765;
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.010421925602951868;
                        } else {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.03022109872549124;
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.763591527938843662) ) ) {
                              result[0] += 0.04848738763591911;
                            } else {
                              result[0] += 0.11141672037079797;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.010681567279170127;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.940579652786255771) ) ) {
                          result[0] += 0.006287103630188322;
                        } else {
                          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.06119865241734246;
                          } else {
                            result[0] += -0.01065937081256743;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.95229363441467374) ) ) {
                    result[0] += -0.060886064544130016;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.09278297424316584) ) ) {
                      result[0] += -0.021548544131231608;
                    } else {
                      result[0] += 0.05530372879490865;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.39368534088134943) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += -0.045049808633819965;
                  } else {
                    result[0] += 0.025382436988482873;
                  }
                } else {
                  result[0] += 0.04397781863275538;
                }
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += 0.07421393545114946;
                  } else {
                    if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.08332798729461517;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += -0.0051900162301879915;
                      } else {
                        result[0] += -0.07014122424308895;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.0017525989259833679;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.222574234008789951) ) ) {
                        result[0] += 0.07438998534417765;
                      } else {
                        result[0] += -0.016567324226900906;
                      }
                    } else {
                      result[0] += -0.07661705754514839;
                    }
                  }
                }
              }
            }
          } else {
            result[0] += -0.027469310623862448;
          }
        }
      } else {
        result[0] += -0.06490163347482068;
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          result[0] += -0.004901416532265;
        } else {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += 0.03305542428648172;
          } else {
            result[0] += -0.0023346386318786193;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += -0.009906909658465654;
        } else {
          result[0] += -0.057532293333022726;
        }
      }
    } else {
      result[0] += -0.034235327836424904;
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[63].missing != -1) && (data[63].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.12467220579493923;
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)56.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)26.50000000000000355) ) ) {
                    result[0] += -0.08914926403728;
                  } else {
                    result[0] += -0.03863528635003832;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -5.4398135865988656e-05;
                    } else {
                      result[0] += -0.05062094139183612;
                    }
                  } else {
                    result[0] += 0.03713105709803486;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)30.50000000000000355) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.01927535820275512;
                  } else {
                    result[0] += 0.039645484252331886;
                  }
                } else {
                  if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.1484347035133911;
                  } else {
                    result[0] += 0.05946082790686801;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.0037344090371340313;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.799905776977539951) ) ) {
                        result[0] += 0.057375935277018675;
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += -0.0025031678342829246;
                        } else {
                          result[0] += -0.06756394640288212;
                        }
                      }
                    } else {
                      result[0] += 0.06989224110631595;
                    }
                  } else {
                    result[0] += -0.06691162246582545;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.239300251007080966) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.56219196319580256) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0794501068675379;
                    } else {
                      result[0] += 0.009115953854210598;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.018504587922376962;
                      } else {
                        result[0] += 0.03555069416364332;
                      }
                    } else {
                      result[0] += -0.06582423159070307;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.634783267974854404) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)250.5000000000000284) ) ) {
                          result[0] += -0.037068750228684215;
                        } else {
                          result[0] += 0.04009138719873277;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.386624813079835761) ) ) {
                          result[0] += 0.03899448148310311;
                        } else {
                          result[0] += -0.016714606261530245;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.04642560099323097;
                        } else {
                          result[0] += -0.06568645744166283;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.924915313720704901) ) ) {
                          result[0] += -0.03169628326577852;
                        } else {
                          result[0] += -0.08850989404547435;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0721170992702911;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.810120582580567294) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.0038476854912811824;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                    result[0] += -0.002730794061778629;
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.03858742568569162;
                    } else {
                      result[0] += -0.10538407276291908;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.03549342106265504;
                } else {
                  result[0] += 0.0030835299548485524;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.28360033035278498) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
                          result[0] += 0.0072223469548505104;
                        } else {
                          result[0] += 0.04230925954773643;
                        }
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)214.5000000000000284) ) ) {
                          result[0] += -0.06463953944079946;
                        } else {
                          result[0] += 0.0172679590751915;
                        }
                      }
                    } else {
                      result[0] += 0.03563990170984927;
                    }
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.07689847393196175;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                          result[0] += -0.005830819287493133;
                        } else {
                          result[0] += -0.05744574672963704;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
                        result[0] += 0.01899790302868958;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                          result[0] += 0.0059784915193075265;
                        } else {
                          result[0] += -0.04114454989651068;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                      result[0] += 0.00338782542920349;
                    } else {
                      result[0] += 0.05072353613238939;
                    }
                  } else {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.011848360734259636;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.004792032094367615;
                      } else {
                        result[0] += -0.08081280625592814;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.09632545528785574;
                } else {
                  result[0] += -0.014272335263130662;
                }
              }
            }
          }
        } else {
          result[0] += -0.061630461623355164;
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.03849744796753107) ) ) {
          result[0] += -0.0008948968945001588;
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
            result[0] += -0.09607984299918193;
          } else {
            result[0] += -0.019332295468488945;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.652390718460083896) ) ) {
      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
        result[0] += 0.0019085825196253165;
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.465247392654419389) ) ) {
          result[0] += -0.004368711740336129;
        } else {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.07503640006332597;
          } else {
            result[0] += -0.025593323199752844;
          }
        }
      }
    } else {
      result[0] += -0.03240633374447923;
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[63].missing != -1) && (data[63].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.12407203595073626;
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.674522399902344638) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)55.50000000000000711) ) ) {
                  if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                    if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += -0.07814911536689374;
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                        result[0] += 0.005808720255507914;
                      } else {
                        result[0] += -0.09280136537261893;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      result[0] += -0.021180246345033647;
                    } else {
                      result[0] += -0.09723424691559865;
                    }
                  }
                } else {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.0003449484497626408;
                    } else {
                      result[0] += -0.05295490134411121;
                    }
                  } else {
                    result[0] += 0.02919769647674691;
                  }
                }
              } else {
                if ( LIKELY( !(data[52].missing != -1) || (data[52].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)25.50000000000000355) ) ) {
                    result[0] += 0.001550030741180691;
                  } else {
                    result[0] += 0.035184057286691287;
                  }
                } else {
                  result[0] += 0.06977631032075657;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)60.50000000000000711) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
                    result[0] += 0.0008491039577229883;
                  } else {
                    result[0] += 0.011428662850447956;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                    result[0] += 0.006451079883930483;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                      result[0] += 0.014296548640344296;
                    } else {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.03457605391875988;
                        } else {
                          if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            result[0] += 0.03940316469291494;
                          } else {
                            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)101.5000000000000142) ) ) {
                              result[0] += 0.014603778218352119;
                            } else {
                              result[0] += -0.09353789787848153;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.06721530563979473;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.66339445114135831) ) ) {
                      result[0] += 0.0474094698511493;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                          result[0] += -0.003555746699692268;
                        } else {
                          result[0] += -0.06913534789438387;
                        }
                      } else {
                        result[0] += -0.09216590251371673;
                      }
                    }
                  } else {
                    result[0] += 0.05827644778389134;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += 0.01297099306316564;
                  } else {
                    result[0] += -0.07797040742055429;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.810120582580567294) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.016848096181786117;
                } else {
                  result[0] += -0.055805596082075;
                }
              } else {
                if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += 0.0324790888670912;
                } else {
                  result[0] += 0.0026495002958843213;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                  if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9648933410644549) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.01364941858328655;
                      } else {
                        if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)214.5000000000000284) ) ) {
                          result[0] += -0.061229214596938114;
                        } else {
                          result[0] += 0.008880522123209748;
                        }
                      }
                    } else {
                      result[0] += 0.032029292139165604;
                    }
                  } else {
                    if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.07309422566785946;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                          result[0] += -0.005039274956746598;
                        } else {
                          result[0] += -0.054741724209692345;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)152.5000000000000284) ) ) {
                        result[0] += 0.01802174764979766;
                      } else {
                        result[0] += -0.01051865535143054;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)118.5000000000000142) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
                      result[0] += 0.005653780281860991;
                    } else {
                      result[0] += 0.04727976512490936;
                    }
                  } else {
                    result[0] += 0.008001006426510525;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.09344906677939828;
                } else {
                  result[0] += -0.014778830540378252;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
              result[0] += 0.014648545859194232;
            } else {
              result[0] += -0.022891800196819495;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0958310763997344;
            } else {
              result[0] += -0.017890829595801776;
            }
          }
        }
      } else {
        result[0] += -0.06181251050179855;
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
          result[0] += 0.04003543393363287;
        } else {
          result[0] += -0.00029892556125608145;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += -0.0010936436953684226;
        } else {
          if ( UNLIKELY( !(data[54].missing != -1) || (data[54].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.009912731934158819;
          } else {
            if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.024773308273015812;
            } else {
              result[0] += -0.060607884989864774;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
          if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)12.00000000000000178) ) ) {
            result[0] += -0.05930288515264588;
          } else {
            result[0] += -0.020518918966450854;
          }
        } else {
          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.06006405901628811;
            } else {
              result[0] += 0.007550874811082202;
            }
          } else {
            result[0] += 0.0542704231120734;
          }
        }
      } else {
        result[0] += -0.05593930502531078;
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( UNLIKELY(  (data[29].missing != -1) && (data[29].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      result[0] += 0.11958544227410106;
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)6.500000000000000888) ) ) {
        if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)12.00000000000000178) ) ) {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)24.50000000000000355) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.75531578063965021) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.055311203002930576) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += -0.06964833507114128;
                  } else {
                    result[0] += 0.0011359458504169395;
                  }
                } else {
                  result[0] += -0.005835378583687885;
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.0723610261803694;
                } else {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.260092735290528232) ) ) {
                      result[0] += -0.11681608084524725;
                    } else {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.02775726854615035;
                      } else {
                        result[0] += -0.06478133300849653;
                      }
                    }
                  } else {
                    result[0] += 0.025004771497842655;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.56219196319580256) ) ) {
                if ( UNLIKELY(  (data[64].missing != -1) && (data[64].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.687107801437378818) ) ) {
                      result[0] += 0.017824747511824782;
                    } else {
                      result[0] += -0.027493159230093475;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += 0.007321621711938392;
                    } else {
                      result[0] += -0.049592792364813586;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.009447671483007203;
                  } else {
                    result[0] += -0.015452534586470435;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.009158881195719934;
                    } else {
                      result[0] += -0.049653329883680396;
                    }
                  } else {
                    result[0] += 0.03023344700876279;
                  }
                } else {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                      result[0] += -0.05051098190767418;
                    } else {
                      result[0] += -0.0009571810573468341;
                    }
                  } else {
                    if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.07587308099119022;
                    } else {
                      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.05508102448082117;
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.547126770019532138) ) ) {
                          result[0] += -0.07875406645026045;
                        } else {
                          result[0] += 0.00912671286815908;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)113.5000000000000142) ) ) {
                  result[0] += -0.017119870462386687;
                } else {
                  result[0] += 0.0268179304082869;
                }
              } else {
                result[0] += -0.0258051351732441;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.09527877378413392;
              } else {
                if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += -0.06753313009774996;
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.10243625053066863;
                  } else {
                    result[0] += -0.03805160262756952;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                result[0] += -0.0011447770957056836;
              } else {
                result[0] += -0.02862647878799466;
              }
            } else {
              result[0] += 0.01399087394558092;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)4.500000000000000888) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                      result[0] += -0.08937204674435115;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += -0.03225631218278657;
                      } else {
                        result[0] += 0.07558210836019784;
                      }
                    }
                  } else {
                    result[0] += 0.017119310950288315;
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.1465614166779023;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.004881381988526279) ) ) {
                      result[0] += -0.04648834172701869;
                    } else {
                      result[0] += 0.10889091888816363;
                    }
                  }
                }
              } else {
                result[0] += 0.07082410158357348;
              }
            } else {
              result[0] += 0.03868798696493142;
            }
          }
        }
      } else {
        result[0] += -0.06343958307710497;
      }
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.078289031982422763) ) ) {
      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)43.50000000000000711) ) ) {
          result[0] += 0.03720793435009779;
        } else {
          result[0] += -0.0004970624295907805;
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.673553824424744096) ) ) {
          result[0] += -0.0010943815956465676;
        } else {
          if ( UNLIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.006806729777427166;
            } else {
              result[0] += -0.08708406564306789;
            }
          } else {
            result[0] += -0.01650384191491683;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439304351806642401) ) ) {
          if ( LIKELY( !(data[63].missing != -1) || (data[63].fvalue <= (double)137.5000000000000284) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.034945011138917792) ) ) {
              result[0] += -0.08445725398854731;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
                result[0] += -0.04422801959914135;
              } else {
                result[0] += 0.044157605159289036;
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.009768066714118501;
            } else {
              result[0] += -0.09676752999214017;
            }
          }
        } else {
          if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.08763049817156378;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                result[0] += 0.06028198068292909;
              } else {
                result[0] += -0.06153318663760578;
              }
            }
          } else {
            if ( LIKELY( !(data[57].missing != -1) || (data[57].fvalue <= (double)3.000000000000000444) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.08560454433545119;
              } else {
                result[0] += 0.01720702601014595;
              }
            } else {
              result[0] += 0.05378738134407096;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.53139376640319913) ) ) {
          result[0] += -0.02935621694373254;
        } else {
          result[0] += -0.08144226122037818;
        }
      }
    }
  }
}

