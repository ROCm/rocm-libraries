
#include "header.h"

void predict_unit3(union Entry* data, double* result) {
  unsigned int tmp;
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
              result[0] += -0.008203638541569745;
            } else {
              result[0] += -0.09964552773419666;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.11382468142981317;
                } else {
                  result[0] += -0.008308143540882159;
                }
              } else {
                result[0] += 0.02221291346382788;
              }
            } else {
              result[0] += 0.0036502665574788666;
            }
          }
        } else {
          result[0] += 1.806622073826993e-05;
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            result[0] += -0.09231768207482775;
          } else {
            result[0] += -0.018712587590772125;
          }
        } else {
          result[0] += 0.16448359029430853;
        }
      }
    } else {
      if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
            result[0] += 0.04236718361033237;
          } else {
            result[0] += -0.0884190476051643;
          }
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.249904870986938921) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.0011024430720417506;
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.11298925026158578;
              } else {
                result[0] += -0.024218718017771427;
              }
            }
          } else {
            result[0] += 0.03809152087823373;
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.11019476756804616;
            } else {
              result[0] += -0.007820009236645756;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                      result[0] += 0.006858680370041727;
                    } else {
                      result[0] += -0.07152475478842256;
                    }
                  } else {
                    result[0] += 0.004125363131953277;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.0008188559675564388;
                  } else {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.07532163743833438;
                    } else {
                      result[0] += 0.011215524838416835;
                    }
                  }
                }
              } else {
                result[0] += 0.035058380272494694;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.24929904937744318) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.11059575898186465;
                    } else {
                      result[0] += 0.008775578775561907;
                    }
                  } else {
                    result[0] += -0.09916253097482053;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += 0.05677587273846483;
                      } else {
                        result[0] += -0.04220950502197479;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += -0.020370717226517163;
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.520232200622559482) ) ) {
                          result[0] += 0.0932326083574399;
                        } else {
                          result[0] += -0.07816530886335114;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.01453892905365446;
                    } else {
                      result[0] += -0.08430304023194823;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.09718405701555358;
                    } else {
                      result[0] += 0.06126512225346097;
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.938867926597595659) ) ) {
                      result[0] += 0.06904161510482364;
                    } else {
                      result[0] += -0.029274301334938708;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.1067318934011554;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                      result[0] += -0.02425903399790659;
                    } else {
                      result[0] += -0.0730073538295793;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += 0.008717950158368282;
                } else {
                  result[0] += -0.05505071180300998;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.01569141117670281;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                            result[0] += -0.02152890335926066;
                          } else {
                            result[0] += 0.06511314938145261;
                          }
                        } else {
                          result[0] += -0.019087375598172952;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
                          result[0] += 0.0004242934632469003;
                        } else {
                          result[0] += 0.047436444613863704;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += -0.08229361125570525;
                    } else {
                      result[0] += 0.08709706129042945;
                    }
                  }
                } else {
                  result[0] += -0.08400515362536697;
                }
              }
            } else {
              result[0] += -0.046496598969233666;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.093674898147583896) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.400584220886231357) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.04826134800572999;
                  } else {
                    result[0] += 0.03633457928615797;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                    result[0] += 0.02750757524694917;
                  } else {
                    result[0] += -0.009608729115108374;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += 0.14083466297239783;
                } else {
                  result[0] += 0.03909088142825793;
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                result[0] += -0.007055512884235424;
              } else {
                result[0] += -0.07887917421816376;
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += 8.234926631482406e-05;
  }
  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
    result[0] += 0.00018312344251628364;
  } else {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.76985812187194913) ) ) {
        if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.05356078379389953;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                  result[0] += -0.09628138725927361;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                    result[0] += -0.08516692429726501;
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0634560385760919;
                    } else {
                      result[0] += -0.06499360038252168;
                    }
                  }
                }
              } else {
                result[0] += -0.004284023083026302;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              result[0] += -0.008388502663044833;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                result[0] += -0.00033101474335466675;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                    result[0] += -0.10127878516175604;
                  } else {
                    result[0] += 0.01867326607950342;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.12298705504478667;
                    } else {
                      result[0] += 0.05338100968003733;
                    }
                  } else {
                    result[0] += 0.010780914931434963;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.06669363937560245;
          } else {
            if ( UNLIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
              result[0] += 0.002758634633551719;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.017386540703135205;
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += -0.032109474746472896;
                  } else {
                    result[0] += 0.008833964562103431;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                  result[0] += -0.044917779064595174;
                } else {
                  result[0] += -0.017762213280637113;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
          result[0] += -0.014408004318111546;
        } else {
          result[0] += 0.06929848501082596;
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.412779808044434482) ) ) {
          result[0] += -0.004804586837130372;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.04839303501820041;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
                  result[0] += -0.01220420221499046;
                } else {
                  result[0] += 0.020067453072556026;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                      result[0] += -0.011571271143341699;
                    } else {
                      result[0] += 0.0031028314036004286;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.022987167130503486;
                    } else {
                      result[0] += -0.00165015143022367;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.00035900373070587763;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.03778258097046512;
                      } else {
                        result[0] += 0.017672094931304654;
                      }
                    } else {
                      result[0] += -0.05588899206022999;
                    }
                  }
                }
              } else {
                result[0] += -0.008968721418595388;
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.05110124964060167;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.901921629905701128) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.44381141662597834) ) ) {
                    result[0] += 0.024073425024892337;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.305786132812500888) ) ) {
                      result[0] += 0.02163724060165173;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.018793233709930773;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.01681862174486159;
                        } else {
                          result[0] += -0.13827452933805823;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.052717685699463779) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.03946203512767339;
                        } else {
                          result[0] += -0.0022836002948819064;
                        }
                      } else {
                        result[0] += 0.02314800294242123;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.897119760513306552) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                          result[0] += 0.01695649177651157;
                        } else {
                          result[0] += -0.028415825614178842;
                        }
                      } else {
                        result[0] += -0.03982472316085446;
                      }
                    }
                  } else {
                    result[0] += -0.02338480951186319;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.624251961708069292) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                      result[0] += 0.0020393357109613226;
                    } else {
                      result[0] += 0.029518206124987612;
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.39944553375244318) ) ) {
                      result[0] += 0.0017177225609351814;
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.007005918697735966;
                      } else {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.022205806250109386;
                        } else {
                          result[0] += 0.018763651650153004;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02311674579307392;
                    } else {
                      result[0] += 0.014191775768119487;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.009967013785964011;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.05327199474033807;
                        } else {
                          result[0] += 0.003581075523407621;
                        }
                      }
                    } else {
                      result[0] += 0.005360072253425342;
                    }
                  }
                }
              } else {
                result[0] += -0.014477794913667023;
              }
            }
          }
        }
      } else {
        result[0] += -0.07305613582296935;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        result[0] += -0.000723349156915781;
      } else {
        if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.62696647644043146) ) ) {
            result[0] += -0.000408083342148853;
          } else {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.005680590675041661;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                result[0] += 0.025113368361233138;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.032497357789388615;
                } else {
                  result[0] += -0.0073284804040883895;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.033043947850606575;
                  } else {
                    result[0] += -0.05917969808850593;
                  }
                } else {
                  result[0] += -0.07765358875584424;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  result[0] += -0.05982391201184069;
                } else {
                  result[0] += 0.04110970448760438;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.010757090459712099;
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                      result[0] += 0.045782897491619755;
                    } else {
                      result[0] += -0.003625515031528589;
                    }
                  }
                } else {
                  result[0] += -0.011671708338324217;
                }
              } else {
                result[0] += -0.004935541830749905;
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
              result[0] += -0.018497774391151467;
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.04906958740059256;
                        } else {
                          result[0] += 0.01841424285918129;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                              result[0] += -0.01705939682465638;
                            } else {
                              result[0] += 0.04097680281208971;
                            }
                          } else {
                            result[0] += 0.060591469903145505;
                          }
                        } else {
                          result[0] += -0.0017374972809303794;
                        }
                      }
                    } else {
                      result[0] += -0.017235916886191603;
                    }
                  } else {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.003976295387454214;
                    } else {
                      result[0] += 0.023100192629126246;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.09543523929562038;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                      result[0] += -0.08032564411530653;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                        result[0] += -0.053627201226626356;
                      } else {
                        result[0] += 0.05206765614334871;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.62696647644043146) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.611996650695801669) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.003991246472609106;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.0031858434574484197;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.196324348449708808) ) ) {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                                result[0] += 0.01038543825606114;
                              } else {
                                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
                                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
                                        result[0] += 0.021679879528437737;
                                      } else {
                                        result[0] += -0.06180958714273585;
                                      }
                                    } else {
                                      result[0] += -0.029795662463219155;
                                    }
                                  } else {
                                    result[0] += 0.02580671045162425;
                                  }
                                } else {
                                  result[0] += -0.0241638220927114;
                                }
                              }
                            } else {
                              result[0] += 0.03249534639775493;
                            }
                          } else {
                            result[0] += -0.0009647538146456864;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.32868957519531428) ) ) {
                              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                                result[0] += -0.03438592288098693;
                              } else {
                                result[0] += -0.0003311760000936343;
                              }
                            } else {
                              result[0] += -0.0006346128202995565;
                            }
                          } else {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.305786132812500888) ) ) {
                                result[0] += -0.0003157383639363469;
                              } else {
                                result[0] += 0.01738942905455438;
                              }
                            } else {
                              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                                result[0] += -0.000396539781283032;
                              } else {
                                result[0] += -0.042508188841229234;
                              }
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                        result[0] += 0.02282549061844838;
                      } else {
                        result[0] += -0.009428728659667214;
                      }
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += -0.03261970295986482;
                          } else {
                            result[0] += 0.011802311782265675;
                          }
                        } else {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.03721893823433668;
                            } else {
                              result[0] += 0.10257355411782323;
                            }
                          } else {
                            result[0] += 0.07693051417006716;
                          }
                        }
                      } else {
                        result[0] += 0.0009943268571651627;
                      }
                    }
                  }
                } else {
                  result[0] += 0.010637016931112712;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += 0.0026262954674676577;
          } else {
            result[0] += -0.21147541070423595;
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.574051618576050693) ) ) {
              result[0] += 0.016110270874987627;
            } else {
              result[0] += 0.13527982528802132;
            }
          } else {
            result[0] += 0.0947135859147483;
          }
        }
      } else {
        result[0] += 0.00718462481714995;
      }
    }
  } else {
    result[0] += -0.016200945980863055;
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
                    result[0] += -0.01386398538794845;
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += 0.08401725224740704;
                    } else {
                      result[0] += -0.04306823225360797;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.966960191726685458) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.701612949371339667) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                            result[0] += -0.00628985209237384;
                          } else {
                            result[0] += 0.011440227944027007;
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.32868957519531428) ) ) {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.650573849678039995) ) ) {
                              result[0] += 0.04037945892064085;
                            } else {
                              result[0] += 0.014568262237807703;
                            }
                          } else {
                            result[0] += -0.024697761913844873;
                          }
                        }
                      } else {
                        result[0] += -0.008496184045763425;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.338562726974488193) ) ) {
                          result[0] += 0.007713250689389595;
                        } else {
                          result[0] += -0.08160016393076791;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                          result[0] += -0.06053130137561529;
                        } else {
                          result[0] += -0.007726298024575737;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
                          result[0] += -0.04114835335646151;
                        } else {
                          result[0] += 0.029267471023012293;
                        }
                      } else {
                        result[0] += 0.004033786863864336;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.869292974472046787) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.164366245269776279) ) ) {
                          result[0] += 0.003520781628007541;
                        } else {
                          result[0] += 0.09720490585812669;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                          result[0] += -0.10085145892328395;
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                            result[0] += 0.16973226588345933;
                          } else {
                            result[0] += 0.07345305148064822;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.257356405258179155) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.48872375488281428) ) ) {
                          result[0] += 0.023252359378412066;
                        } else {
                          result[0] += 0.09063257741584434;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                            result[0] += 0.02783945149781312;
                          } else {
                            result[0] += -0.009233165940983706;
                          }
                        } else {
                          result[0] += 0.1331034753541495;
                        }
                      }
                    } else {
                      result[0] += 0.06606416775900066;
                    }
                  } else {
                    result[0] += -0.043203923746785555;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.76988101005554288) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.551017761230469638) ) ) {
                        result[0] += -0.006166110730203028;
                      } else {
                        result[0] += -0.04196851274040528;
                      }
                    } else {
                      result[0] += -0.10789615138453446;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.589327573776246005) ) ) {
                        result[0] += -0.014105922477698325;
                      } else {
                        result[0] += 0.03392288085236513;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.06071434108701194;
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.01634240150451749) ) ) {
                                result[0] += 0.003126279896026916;
                              } else {
                                result[0] += -0.023677883418741393;
                              }
                            } else {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
                                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                                  result[0] += -0.006682378981400345;
                                } else {
                                  result[0] += -0.07220660223450541;
                                }
                              } else {
                                result[0] += -0.13731997546104352;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.57691621780395685) ) ) {
                              result[0] += -1.8065886757632446e-05;
                            } else {
                              result[0] += 0.0681693855630171;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.087486267089845526) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
                            result[0] += -0.0015541959445369267;
                          } else {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                              result[0] += -0.09583208554513888;
                            } else {
                              result[0] += 0.0639156281877634;
                            }
                          }
                        } else {
                          result[0] += 0.009539906222164423;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += 0.014443894623510048;
                    } else {
                      result[0] += -0.030276961118492957;
                    }
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.00138011677566151;
                    } else {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                        result[0] += -0.08937724565940908;
                      } else {
                        result[0] += -0.03881839001324884;
                      }
                    }
                  }
                } else {
                  result[0] += -0.003142323094729392;
                }
              } else {
                result[0] += 0.058634661310427286;
              }
            }
          } else {
            result[0] += -0.022763568438225686;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.001777980931366883;
              } else {
                result[0] += -0.12442532621147498;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.06468080412492004;
                } else {
                  result[0] += -0.0008819153351622193;
                }
              } else {
                result[0] += 0.014275650434250381;
              }
            }
          } else {
            result[0] += 0.05347189050358027;
          }
        }
      } else {
        result[0] += -0.035294544061785504;
      }
    } else {
      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.00344299197586027;
      } else {
        result[0] += -0.01796042033522847;
      }
    }
  } else {
    result[0] += 6.023859096723389e-05;
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.00038046047140596916;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
            result[0] += -0.0013792520621758566;
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.005359392203111164;
                  } else {
                    result[0] += 0.07356794915430862;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.303973913192749912) ) ) {
                    result[0] += -0.022179376703240883;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                        result[0] += 0.026303169731994192;
                      } else {
                        result[0] += -0.03905450487414951;
                      }
                    } else {
                      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -0.00942967605498318;
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.668153762817383701) ) ) {
                          result[0] += 0.005284254181637122;
                        } else {
                          result[0] += 0.016761385309837517;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.005980861967646627;
                } else {
                  result[0] += -0.0019288443372388023;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += 0.006221814556241174;
              } else {
                result[0] += -0.013290288178648838;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.797939777374268466) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  result[0] += 0.01259318877400815;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.052006508506973415;
                  } else {
                    result[0] += -0.015888020286654772;
                  }
                }
              } else {
                result[0] += -0.015400335033097163;
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.594409704208374912) ) ) {
                      result[0] += -0.008997539008393122;
                    } else {
                      result[0] += -0.04205107456910638;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                        if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.01007813368839913;
                        } else {
                          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                            result[0] += 0.005943978776842144;
                          } else {
                            result[0] += 0.03456014872362891;
                          }
                        }
                      } else {
                        result[0] += 0.0027279377716501857;
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.009390396878590564;
                      } else {
                        result[0] += 0.0018097376339531028;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.701612949371339667) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                        result[0] += 0.00012708777086158852;
                      } else {
                        result[0] += 0.04234314215924749;
                      }
                    } else {
                      result[0] += 0.021585847558042368;
                    }
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += 0.00906424338627322;
                      } else {
                        result[0] += -0.00865198486244015;
                      }
                    } else {
                      result[0] += -0.014824658986896545;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.045184262070138224;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.725620865821838823) ) ) {
                        result[0] += -0.032494441604861306;
                      } else {
                        result[0] += 0.009504514342899319;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.00923028629600304;
                    } else {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.0007437224385060555;
                      } else {
                        result[0] += -0.03860518505419644;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.0062903041171616094;
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.928530216217041904) ) ) {
                            result[0] += -0.033017803738704596;
                          } else {
                            result[0] += -0.15622554832069996;
                          }
                        } else {
                          result[0] += -0.007612111272956647;
                        }
                      }
                    } else {
                      result[0] += -0.002940485239791071;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.008541161078667505;
                      } else {
                        result[0] += 0.023067618506582403;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.009600256841043358;
                        } else {
                          result[0] += -0.006342575187031062;
                        }
                      } else {
                        result[0] += 0.00778540742629899;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  result[0] += -0.013030908872800985;
                } else {
                  result[0] += 0.010131781342558325;
                }
              } else {
                result[0] += 0.020523215713985733;
              }
            } else {
              result[0] += -0.00889432867546753;
            }
          }
        }
      }
    } else {
      result[0] += 0.11303574083455858;
    }
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.901921629905701128) ) ) {
              result[0] += 0.14564889258380767;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.197173833847046787) ) ) {
                result[0] += 0.006994341663721997;
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.032642068244210926;
                } else {
                  result[0] += 0.1497005065682959;
                }
              }
            }
          } else {
            result[0] += 0.06760434076236074;
          }
        } else {
          result[0] += 0.008490193143149524;
        }
      } else {
        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += 0.031691772784644336;
          } else {
            result[0] += -0.01814033198848344;
          }
        } else {
          result[0] += -0.09477141432986068;
        }
      }
    } else {
      result[0] += -0.11361407423049828;
    }
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.00033434049636715205;
      } else {
        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.300811052322388583) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.668367385864259589) ) ) {
                              result[0] += -0.04685055137620807;
                            } else {
                              result[0] += 0.01927858620055786;
                            }
                          } else {
                            result[0] += 0.0459624832277946;
                          }
                        } else {
                          result[0] += -0.06535930557227111;
                        }
                      } else {
                        result[0] += 0.05884879277618086;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                        result[0] += -0.038080116522469565;
                      } else {
                        result[0] += 0.056846427592400196;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                      result[0] += 0.012547118713403369;
                    } else {
                      result[0] += -0.010213175685644428;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.511434078216553178) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.08042614347059854;
                      } else {
                        result[0] += 0.031779641616052394;
                      }
                    } else {
                      result[0] += 0.10024850902953036;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.045472116594562906;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.774795055389405185) ) ) {
                        result[0] += -0.07281567846943246;
                      } else {
                        result[0] += -0.002334338642844266;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                  result[0] += -0.0283635759768397;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.274755001068116123) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                          result[0] += 0.11641804405817005;
                        } else {
                          result[0] += 0.019071259963263712;
                        }
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.00021941372469633572;
                        } else {
                          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.925687789916993964) ) ) {
                              result[0] += 0.007848136706707472;
                            } else {
                              result[0] += -0.03159471499679698;
                            }
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                              result[0] += -0.057411201446984084;
                            } else {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
                                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                                  result[0] += -0.05129082077269986;
                                } else {
                                  result[0] += -0.008598674037540445;
                                }
                              } else {
                                result[0] += 0.014495230562039033;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += -0.039517538188726366;
                      } else {
                        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                          result[0] += 0.018924453751337177;
                        } else {
                          result[0] += 0.0027149931505281743;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.013363724835854442;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.592359304428101474) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                  result[0] += -0.010203258070261413;
                } else {
                  result[0] += 0.014275162344231604;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += -0.0010638846058530918;
                } else {
                  result[0] += -0.018434538784078667;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.724856853485109198) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                  result[0] += -0.016171752133974638;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.01780356816485355;
                          } else {
                            result[0] += 0.050734345436573874;
                          }
                        } else {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.023372147732865514;
                          } else {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.15100884437561124) ) ) {
                              result[0] += 0.02155670889880117;
                            } else {
                              result[0] += -0.07461607668338886;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += 0.003616962276481372;
                        } else {
                          result[0] += -0.0249456478298081;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.701225757598877397) ) ) {
                        result[0] += -0.021305472770607056;
                      } else {
                        result[0] += 0.01596985841015663;
                      }
                    }
                  } else {
                    result[0] += -0.013867069249680182;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.354653358459474433) ) ) {
                  result[0] += 0.025656682763301682;
                } else {
                  result[0] += -0.001095181366485969;
                }
              }
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.048414796753097296;
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.02902675185000213;
                  } else {
                    result[0] += 0.0015475866562124217;
                  }
                }
              } else {
                result[0] += -0.007305968262304341;
              }
            }
          }
        } else {
          result[0] += 0.0008902871398618027;
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              result[0] += 0.005749101714175953;
            } else {
              result[0] += 0.05509403051519738;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.07534529373925751;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.847873449325562412) ) ) {
                  result[0] += 0.025838335897803644;
                } else {
                  result[0] += 0.09280456798868635;
                }
              } else {
                result[0] += 0.005543172210834391;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += 0.021459578262635594;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
              result[0] += -0.01340447126588475;
            } else {
              result[0] += -0.07202015668498761;
            }
          }
        }
      } else {
        result[0] += -0.10856803249705288;
      }
    }
  } else {
    result[0] += 0.11303574083455858;
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.00034566769332265756;
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += -0.011698780059848674;
                      } else {
                        result[0] += 0.10143904596628886;
                      }
                    } else {
                      result[0] += -0.12234092072679881;
                    }
                  } else {
                    result[0] += 0.06711948676620333;
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                    result[0] += 0.027713414592495572;
                  } else {
                    result[0] += 0.007588874226067598;
                  }
                }
              } else {
                result[0] += -0.0010337318471777823;
              }
            } else {
              result[0] += -0.001595190578658136;
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += 0.09897338111198517;
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.00505223530591774;
                      } else {
                        result[0] += -0.00871640639329737;
                      }
                    } else {
                      result[0] += 0.004744238890157685;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                    result[0] += -0.0003147143435207043;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.03127310683717799;
                      } else {
                        result[0] += 0.022781998960356403;
                      }
                    } else {
                      result[0] += 0.011506992743490133;
                    }
                  }
                }
              } else {
                result[0] += -0.008156162752888703;
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.0018830173164350904;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.278613805770874912) ) ) {
                  result[0] += 0.011338886047404968;
                } else {
                  result[0] += 0.0838102488521818;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.994848489761353427) ) ) {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0015980272697952728;
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += 0.010468111838322358;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.23602247238159357) ) ) {
                      result[0] += -0.009533167879545775;
                    } else {
                      result[0] += -0.028443695767664354;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                    result[0] += -0.04759816042984756;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                      result[0] += -0.03601102688428997;
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.011530340448392157;
                        } else {
                          result[0] += 0.04729387468088156;
                        }
                      } else {
                        result[0] += -0.001208775105002889;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
                        result[0] += 0.0015126003979557502;
                      } else {
                        result[0] += -0.00935096878774322;
                      }
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.004489931035886754;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.003911584620386213;
                          } else {
                            result[0] += 0.07410427636335751;
                          }
                        } else {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.547126770019532138) ) ) {
                            result[0] += -0.03326979758357962;
                          } else {
                            result[0] += 0.027510954145608155;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                          result[0] += 0.003327196019047003;
                        } else {
                          result[0] += -0.01623014610803856;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                          result[0] += -0.02314219656427127;
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                            result[0] += 0.03912863543961633;
                          } else {
                            result[0] += -0.013724794112953643;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.0046311426696703396;
                    }
                  }
                } else {
                  result[0] += -0.09773380613854286;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                result[0] += -0.027859741870611705;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.06435945997043459;
                    } else {
                      result[0] += -0.04757203543173627;
                    }
                  } else {
                    if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.03484397073542962;
                    } else {
                      result[0] += -0.06192595569645459;
                    }
                  }
                } else {
                  result[0] += -0.04642348814242481;
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.309873342514038974) ) ) {
                    result[0] += -0.022681797121583733;
                  } else {
                    result[0] += -0.07751853902999899;
                  }
                } else {
                  result[0] += -0.005877939420996059;
                }
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                  result[0] += 0.0021022146856765184;
                } else {
                  result[0] += -0.031704674910576006;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
            result[0] += 0.007016682129195296;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.07160834657743945;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.837577581405640537) ) ) {
                  result[0] += 0.02407323558312429;
                } else {
                  result[0] += 0.09463408046430162;
                }
              } else {
                result[0] += 0.00526403902005098;
              }
            }
          }
        } else {
          result[0] += -0.009383699700971566;
        }
      } else {
        result[0] += -0.10525127521821867;
      }
    }
  } else {
    result[0] += 0.11303574083455858;
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
      result[0] += -0.00020161123896621433;
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
          result[0] += 0.0019671191650432207;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.26534414291382014) ) ) {
              result[0] += 0.0009296907765183335;
            } else {
              result[0] += -0.02737614420375774;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.043312669253552434;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.026457641116780107;
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.01829288124168704;
                  } else {
                    result[0] += 0.010554407007802915;
                  }
                }
              }
            } else {
              result[0] += -0.048808311155887366;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
                  result[0] += 0.019934661651580282;
                } else {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += -0.05786949848542178;
                          } else {
                            result[0] += 0.03506737899400588;
                          }
                        } else {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.03383229324080608;
                          } else {
                            result[0] += -0.00676691953900198;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.314458370208742011) ) ) {
                            result[0] += -0.018443086748096964;
                          } else {
                            result[0] += 0.010271527449827761;
                          }
                        } else {
                          result[0] += -0.07158440593102568;
                        }
                      }
                    } else {
                      result[0] += -0.028758349678916335;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.018293692833135877;
                      } else {
                        result[0] += -0.021257140210942452;
                      }
                    } else {
                      result[0] += 0.004026359369404819;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.023400783538819248) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += 0.005938820806940616;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += 0.011055758479459095;
                        } else {
                          result[0] += -0.007069444752763933;
                        }
                      } else {
                        result[0] += -0.1511332944586235;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.0015576180314620096;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.029260324373548426;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.05716129262984218;
                              } else {
                                result[0] += -0.011665094821051755;
                              }
                            } else {
                              result[0] += 0.00436126606726084;
                            }
                          } else {
                            result[0] += 0.009275925637375781;
                          }
                        } else {
                          if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.000264869546091438;
                          } else {
                            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.02193641662597834) ) ) {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                                  result[0] += -0.030828706949282025;
                                } else {
                                  result[0] += 0.010462408244244556;
                                }
                              } else {
                                result[0] += 0.02010267201606208;
                              }
                            } else {
                              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                                result[0] += -0.04500322393429154;
                              } else {
                                result[0] += 0.004278512468311768;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.009091479105838308;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                result[0] += -0.011788802666112923;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += 0.019175263587467753;
                  } else {
                    result[0] += -0.007517117112456647;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
                    result[0] += 0.007307479819036117;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      result[0] += 0.01759819082081273;
                    } else {
                      result[0] += 0.045605738242720455;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                result[0] += -0.007165036960526947;
              } else {
                result[0] += 0.001804386901813638;
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.438968896865845615) ) ) {
                result[0] += -0.022742365003271522;
              } else {
                result[0] += 0.06058039292834701;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += -0.006438708830346389;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.022670897814547945;
            } else {
              result[0] += 0.0037766827461348886;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
            result[0] += 0.005397138806878851;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.06880826751853385;
            } else {
              result[0] += -0.061644676592435256;
            }
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.074195146560669833) ) ) {
            result[0] += 0.12003212844843283;
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += 0.06668221502875633;
            } else {
              if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.03470885342926596;
              } else {
                result[0] += 0.00508969642559471;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.028216775959499552;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.169590950012207919) ) ) {
              result[0] += -0.027916050411338447;
            } else {
              result[0] += -0.17301356396688847;
            }
          }
        } else {
          result[0] += 0.0029597324170579148;
        }
      }
    } else {
      result[0] += -0.1014562737835832;
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += -0.00023113178556176059;
      } else {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.003507989656064208;
                } else {
                  result[0] += 0.028311239615129047;
                }
              } else {
                result[0] += -0.006903033815552691;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.597218394279480425) ) ) {
                result[0] += -0.04746262407503154;
              } else {
                result[0] += 0.017724302086988504;
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.038464051423304196;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.314458370208742011) ) ) {
                  result[0] += 0.005247875556616472;
                } else {
                  if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.031958336809929475;
                  } else {
                    result[0] += -0.0019296976373138345;
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += -0.026921336971475054;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.017228473685397476;
                    } else {
                      result[0] += 0.009275462941969709;
                    }
                  } else {
                    result[0] += -0.04148999788216434;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
                    result[0] += 0.018283047842671814;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.01448351614736248;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                            result[0] += -0.008121450420769139;
                          } else {
                            result[0] += -0.03291275630149713;
                          }
                        }
                      } else {
                        result[0] += 0.0038679542216333556;
                      }
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.586156606674195224) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += -0.06910614560586686;
                            } else {
                              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.075335502624512607) ) ) {
                                result[0] += 0.034461369143358594;
                              } else {
                                result[0] += 0.005057039269157259;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.005346541880604058;
                            } else {
                              result[0] += -0.019515502906751206;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += -0.030514233091667682;
                          } else {
                            result[0] += -0.007962661577406955;
                          }
                        }
                      } else {
                        result[0] += -0.026090666731920554;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.023400783538819248) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.0007204340286337129;
                        } else {
                          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                              result[0] += 0.0024751749273116165;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                                result[0] += -0.02386010601739684;
                              } else {
                                result[0] += -0.06355661252109511;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
                                result[0] += -0.07735411220021328;
                              } else {
                                result[0] += -0.010226366621703686;
                              }
                            } else {
                              result[0] += 0.004879218180972768;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.10488171713404162;
                      }
                    } else {
                      result[0] += 0.0035914062442181866;
                    }
                  } else {
                    result[0] += 0.008487644935830272;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  result[0] += -0.010967743479660085;
                } else {
                  result[0] += 0.008393275670136528;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.0015023571300707755;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
                      result[0] += 0.04164791548555163;
                    } else {
                      result[0] += -0.004792834000711539;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.791781663894654208) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.051500320434572089) ) ) {
                        result[0] += -0.02801185773036223;
                      } else {
                        result[0] += -0.056480412626803825;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.797939777374268466) ) ) {
                          result[0] += -0.025513597781826488;
                        } else {
                          result[0] += -0.10405382860755819;
                        }
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                            result[0] += 0.035524087914308715;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                              result[0] += -0.0858934746317299;
                            } else {
                              result[0] += 0.022307984765573163;
                            }
                          }
                        } else {
                          result[0] += -0.008333121426778153;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += 0.00022704962069290235;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.006634288418504349;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += 0.034452020935540166;
                  } else {
                    result[0] += -0.03914512676905002;
                  }
                } else {
                  result[0] += 0.010480836157907764;
                }
              } else {
                result[0] += 0.0035564308059732406;
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.11227477241476745;
    }
  } else {
    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.139882326126099521) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
            result[0] += 0.006832304551771602;
          } else {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)4.500000000000000888) ) ) {
              result[0] += 0.01963885099154637;
            } else {
              result[0] += 0.06600414488102056;
            }
          }
        } else {
          result[0] += -0.006790764368942004;
        }
      } else {
        result[0] += 0.10177169788526118;
      }
    } else {
      result[0] += -0.09934821283252995;
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
            result[0] += -0.00025966277358029994;
          } else {
            if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.422742605209351474) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                    result[0] += 0.028185861990752743;
                  } else {
                    result[0] += -0.024976868205214282;
                  }
                } else {
                  result[0] += -0.02458872920108112;
                }
              } else {
                result[0] += 0.02790963773298348;
              }
            } else {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += -0.021951798646800743;
              } else {
                result[0] += -0.061954730876843736;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.740319490432739702) ) ) {
            result[0] += 0.0108136667869322;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.035206551331297804;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.158509254455567294) ) ) {
                        result[0] += -0.029634825633372244;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                            result[0] += 0.012922934629689499;
                          } else {
                            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.0043128693678848;
                            } else {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                                result[0] += 0.030897701218881912;
                              } else {
                                result[0] += 0.08461079796435693;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.01349988240272204;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                        result[0] += 0.05669617579758181;
                      } else {
                        result[0] += -0.022036913399617036;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.012022389486012206;
                      } else {
                        result[0] += 0.10751902405428858;
                      }
                    } else {
                      result[0] += -0.055874712167925916;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.0035333299011318167;
                        } else {
                          result[0] += 0.005376901163255683;
                        }
                      } else {
                        result[0] += 0.014733880718126521;
                      }
                    } else {
                      result[0] += -0.0036907186508469596;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
                        result[0] += 0.011975708804150062;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05539035797119318) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.259080410003662998) ) ) {
                              result[0] += -0.0029145335597559135;
                            } else {
                              result[0] += -0.05048386701724897;
                            }
                          } else {
                            result[0] += -0.05510694382243905;
                          }
                        } else {
                          result[0] += 0.011419364234695207;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                          result[0] += -0.018509333821254484;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.010103411579507782;
                          } else {
                            result[0] += -0.07149326972263902;
                          }
                        }
                      } else {
                        result[0] += -0.004717312372989488;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.512487888336182529) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY(  (data[44].missing != -1) && (data[44].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                        result[0] += -0.004036049056568951;
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.349460363388062412) ) ) {
                            result[0] += 0.007249154673354394;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                              result[0] += -0.04469925892355521;
                            } else {
                              result[0] += -0.007502696089622368;
                            }
                          }
                        } else {
                          result[0] += -0.0460434432693973;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.004478590058067039;
                      } else {
                        result[0] += -0.037797190818263535;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                      result[0] += -0.02622890782136277;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                        result[0] += 0.0019470690931220104;
                      } else {
                        result[0] += 0.01194965596559691;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.525927901268005815) ) ) {
                      result[0] += -0.011465620837223342;
                    } else {
                      result[0] += 0.05755391989781851;
                    }
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += 0.01188656727543776;
                        } else {
                          result[0] += 0.05545863223217824;
                        }
                      } else {
                        result[0] += 0.003180991126619734;
                      }
                    } else {
                      result[0] += -0.0016698787461150739;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.10140453712011765;
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.038159370422365058) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            result[0] += -0.011496170329565849;
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              result[0] += 0.008756495470624513;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.02565863070301211;
              } else {
                result[0] += 0.0820326981172822;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
            result[0] += 0.11767808663160383;
          } else {
            result[0] += 0.006284553055993298;
          }
        }
      } else {
        result[0] += -0.09557914767164827;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
      result[0] += 0.0006416328784141772;
    } else {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)47227863040.00000763) ) ) {
        result[0] += 0.02091793490307646;
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.848108768463135654) ) ) {
            result[0] += -0.04631000726300995;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += -0.16911235114295708;
            } else {
              result[0] += -0.06899977919214623;
            }
          }
        } else {
          result[0] += -0.017536806879793202;
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)12.00000000000000178) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.122375011444092685) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.0003421083042361779;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.048955064449610536;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      result[0] += 0.0072553440926559735;
                    } else {
                      result[0] += 0.03687784745904372;
                    }
                  } else {
                    result[0] += -0.012241845055631174;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.058321010463380354;
                  } else {
                    result[0] += -0.014822736963882808;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.02565013035894618;
                  } else {
                    result[0] += -0.005390148390023679;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.0015814070418119367;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                      result[0] += 0.018526997870143346;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                        result[0] += -0.036476725805535126;
                      } else {
                        result[0] += 0.021129854154883318;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                    result[0] += -0.010306050690234443;
                  } else {
                    result[0] += -0.00019846519358506163;
                  }
                }
              } else {
                result[0] += 0.008514986926153515;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.579839229583741123) ) ) {
              if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.1483068753533747;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                  result[0] += 0.0497181358511961;
                } else {
                  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.1511337881550689;
                  } else {
                    result[0] += 0.00870988141559263;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.10410410186590341;
                    } else {
                      result[0] += 0.019257326539363646;
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.09946919656971953;
                      } else {
                        result[0] += -0.004264621353571187;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.740319490432739702) ) ) {
                        result[0] += -0.03963841268039033;
                      } else {
                        result[0] += 0.0468193275909034;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.549068689346314365) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                          result[0] += 0.07086927646198915;
                        } else {
                          if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.1096833727641551;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                              result[0] += -0.09030305611060378;
                            } else {
                              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                                result[0] += 0.0625714144513442;
                              } else {
                                result[0] += -0.06243196780654199;
                              }
                            }
                          }
                        }
                      } else {
                        result[0] += -0.14293401764414446;
                      }
                    } else {
                      result[0] += -0.13775481071913787;
                    }
                  } else {
                    result[0] += 0.11784367051980563;
                  }
                }
              } else {
                result[0] += 0.037717099999934195;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.966960191726685458) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.95229363441467374) ) ) {
                result[0] += -0.12795744418010546;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.561026811599732333) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                      result[0] += 0.05922130579329681;
                    } else {
                      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                        result[0] += 0.05694904373323854;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.198464870452881303) ) ) {
                            result[0] += 0.13693834829437004;
                          } else {
                            result[0] += -0.05130115600287658;
                          }
                        } else {
                          result[0] += -0.12727773190930322;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.12333895036560921;
                  }
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.13274036491696903;
                  } else {
                    if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.15400157380931775;
                    } else {
                      result[0] += 0.04432882829095729;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                  result[0] += 0.14221449476220574;
                } else {
                  result[0] += 0.0022233872690984703;
                }
              } else {
                result[0] += -0.12787451756149676;
              }
            }
          }
        }
      } else {
        result[0] += 0.10616331318308186;
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.038159370422365058) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
              result[0] += 0.002167615389355837;
            } else {
              result[0] += -0.03343368749787926;
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              result[0] += 0.008289257483612785;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.027232469993182615;
              } else {
                result[0] += 0.07889472505964323;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.15100884437561124) ) ) {
            result[0] += 0.11542374922728123;
          } else {
            result[0] += 0.005553516446978088;
          }
        }
      } else {
        result[0] += -0.08731595994087421;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
      result[0] += 0.0013644630532235274;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.848108768463135654) ) ) {
            result[0] += -0.04538649961762824;
          } else {
            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
              result[0] += -0.1693762525007942;
            } else {
              result[0] += -0.06439334574444903;
            }
          }
        } else {
          result[0] += -0.016636577883693902;
        }
      } else {
        result[0] += 0.021983398866442645;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        result[0] += -0.00036295969581372537;
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.740319490432739702) ) ) {
            result[0] += 0.01546597747395427;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.673758983612062323) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.680079460144043857) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                      result[0] += 0.03994197827959946;
                    } else {
                      result[0] += 0.011109738387285982;
                    }
                  } else {
                    result[0] += -0.03174354134480683;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                    result[0] += -0.013433481684643873;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += 0.005086214023268842;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.050306260876231225;
                        } else {
                          result[0] += -0.005964140274790964;
                        }
                      } else {
                        result[0] += -0.09659304148032846;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.008667082898106028;
                } else {
                  if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.05740288671126299;
                  } else {
                    result[0] += -0.01338256961546901;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.189540147781372958) ) ) {
                    result[0] += 0.014061642883751658;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0001238058888935699;
                    } else {
                      result[0] += 0.019486414814672774;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.437634944915773261) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            result[0] += -0.029122016523115097;
                          } else {
                            result[0] += -0.09469836644136374;
                          }
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                            result[0] += -0.01632899364149128;
                          } else {
                            result[0] += 0.03562283869657815;
                          }
                        }
                      } else {
                        result[0] += -0.003988765357769061;
                      }
                    } else {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.0032791486976663196;
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
                              result[0] += 0.08012689173129556;
                            } else {
                              result[0] += 0.023434860950154485;
                            }
                          }
                        } else {
                          result[0] += 0.0008047182605494391;
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.08438852878588733;
                            } else {
                              result[0] += 0.003143844469991823;
                            }
                          } else {
                            result[0] += 0.010561748201031833;
                          }
                        } else {
                          result[0] += -0.05747314470182518;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.07144080971346602;
                  }
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.03535687414270191;
                    } else {
                      result[0] += -0.005166884846346598;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
                      result[0] += 0.0003896307423188445;
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                        result[0] += -3.266493330433032e-05;
                      } else {
                        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.035036703630256065;
                          } else {
                            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.016086410269486706;
                            } else {
                              result[0] += 0.037838739484468684;
                            }
                          }
                        } else {
                          result[0] += 0.03617125171271384;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.349460363388062412) ) ) {
                    result[0] += -0.013986230757932991;
                  } else {
                    result[0] += 0.006896538667707783;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.449861526489258257) ) ) {
              result[0] += -0.0298688011438806;
            } else {
              result[0] += 0.017918413892261557;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                result[0] += 0.02108771374882844;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.06371751302261379;
                } else {
                  result[0] += -0.014736382662624276;
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.777633190155030185) ) ) {
                  result[0] += 0.0042796281270990905;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.0036601990385224967;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)9.833202838897706854) ) ) {
                      result[0] += -0.030497746441061176;
                    } else {
                      result[0] += 0.15421682974162076;
                    }
                  }
                }
              } else {
                result[0] += 0.0030759130883701116;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.038159370422365058) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
              result[0] += 0.0021665559796899213;
            } else {
              result[0] += -0.03330030548057574;
            }
          } else {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
              result[0] += 0.007922036698199944;
            } else {
              result[0] += 0.03941141601874598;
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.15100884437561124) ) ) {
            result[0] += 0.11525455609520828;
          } else {
            result[0] += 0.004899443395307743;
          }
        }
      } else {
        result[0] += -0.08442187072863462;
      }
    }
  } else {
    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
      result[0] += 0.001759477420829794;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
              result[0] += -0.0407693623458559;
            } else {
              result[0] += -0.14128771867884055;
            }
          } else {
            result[0] += -0.018189605418410238;
          }
        } else {
          result[0] += -0.01580779256797125;
        }
      } else {
        result[0] += 0.021695049922082044;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.00035850770621066947;
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)13.86392068862915217) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
              result[0] += -0.011541507080921661;
            } else {
              if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                      result[0] += 0.014806615292026854;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.0013512334328542107;
                      } else {
                        result[0] += -0.02185861694002823;
                      }
                    }
                  } else {
                    result[0] += -0.030556272652211038;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                      result[0] += 0.0021178101110151527;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0023572452680460706;
                      } else {
                        result[0] += -0.05569899140816358;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.566809177398682529) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += -0.0034575005375508923;
                          } else {
                            result[0] += 0.017441733771032444;
                          }
                        } else {
                          result[0] += 0.030400984803350486;
                        }
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                            result[0] += -0.08590576749371566;
                          } else {
                            result[0] += 0.026962664500673545;
                          }
                        } else {
                          result[0] += -0.022502109536004542;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                        result[0] += -0.019929694229370628;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                            result[0] += 0.03349937197797169;
                          } else {
                            result[0] += 0.07135969752547786;
                          }
                        } else {
                          result[0] += -0.010913591732842858;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.004729962180402507;
                } else {
                  result[0] += 0.01567634987278776;
                }
              }
            }
          } else {
            result[0] += -0.036083952889204;
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.05849269194911925;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7.87937188148498624) ) ) {
                result[0] += -0.01134189869231733;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.2795047902527667;
                } else {
                  result[0] += -0.014519095082069493;
                }
              }
            }
          } else {
            result[0] += 0.005465247111591979;
          }
        }
      } else {
        result[0] += -0.008584641096022515;
      }
    } else {
      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.66339445114135831) ) ) {
            result[0] += 0.028443180886684502;
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              result[0] += 0.006343609703933752;
            } else {
              result[0] += -0.027744845448921435;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += -0.0031299069116274727;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.003838300704956943) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 8.235252893043613e-05;
              } else {
                result[0] += -0.034696387257260965;
              }
            } else {
              result[0] += 0.007991867158107329;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += -0.017009225550401547;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                  result[0] += 0.019957868725310645;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                    result[0] += 0.010029565489765583;
                  } else {
                    result[0] += -0.01014981966582642;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                result[0] += -0.04550819981168778;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12553024291992365) ) ) {
                  result[0] += 0.01699661844298934;
                } else {
                  result[0] += -0.01900446333956024;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.47223544120788663) ) ) {
              if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                result[0] += -0.004180304996151042;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                  result[0] += -0.037056479362709886;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.03667795102388625;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
                        result[0] += -0.0030747846089890086;
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.038724927457197605;
                        } else {
                          result[0] += -0.015573029124027027;
                        }
                      }
                    } else {
                      result[0] += -0.03583670383193996;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.02494289021543984;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += 0.0027962381639044685;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.453179836273194248) ) ) {
                result[0] += -0.0018329282984879806;
              } else {
                result[0] += -0.03349956447895542;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80658149719238459) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.329718828201294833) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                  result[0] += 0.08636752671245945;
                } else {
                  result[0] += -0.0036146850539692703;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                  result[0] += -0.03464763365336445;
                } else {
                  result[0] += 0.042660387336620906;
                }
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.382196187973023349) ) ) {
                if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.008524185522393962;
                } else {
                  result[0] += 0.0290449792971336;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                      result[0] += -0.019033741014565966;
                    } else {
                      result[0] += 0.05442327446086044;
                    }
                  } else {
                    result[0] += -0.027607646507178348;
                  }
                } else {
                  result[0] += 0.09039407270626887;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
      result[0] += -0.00017147459948911543;
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.680079460144043857) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.350240230560303178) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
                      result[0] += -0.04615168373231782;
                    } else {
                      result[0] += -0.13950955558382022;
                    }
                  } else {
                    result[0] += -0.009492672569672985;
                  }
                } else {
                  result[0] += 0.004713621247132635;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.400584220886231357) ) ) {
                    result[0] += -0.035339227122348424;
                  } else {
                    result[0] += -0.000652074436227852;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
                    result[0] += -0.01674214774354564;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                      result[0] += -0.04359528230075446;
                    } else {
                      result[0] += -0.1402149685656747;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.019594992456928157;
            }
          } else {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              result[0] += -0.014831966239047834;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.37995386123657404) ) ) {
                  result[0] += -0.028914454773269194;
                } else {
                  result[0] += 0.022821249455843847;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.012465653824322418;
                  } else {
                    result[0] += 0.03535247664710619;
                  }
                } else {
                  result[0] += -0.011915684748014535;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.726826429367066318) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += 0.0012968546674252193;
                } else {
                  result[0] += 0.03500099599288378;
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                    result[0] += -0.11030680695493056;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                        result[0] += -0.00246832120410148;
                      } else {
                        result[0] += 0.0699405987258479;
                      }
                    } else {
                      result[0] += -0.054499383588034536;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.068990230560303623) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.010748206985423763;
                      } else {
                        result[0] += 0.033036279916428844;
                      }
                    } else {
                      result[0] += -0.024751346660954535;
                    }
                  } else {
                    result[0] += 0.048157336245386824;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                result[0] += -0.08521519931237925;
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81804704666137873) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += -0.059363133248940894;
                    } else {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.463808774948121005) ) ) {
                        result[0] += -0.09002570754765102;
                      } else {
                        result[0] += 0.07946887113881028;
                      }
                    }
                  } else {
                    result[0] += 0.022535684850576887;
                  }
                } else {
                  result[0] += 0.030428041580072347;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.067782521247864214) ) ) {
              result[0] += 0.04096189375053802;
            } else {
              result[0] += -0.016665809399949296;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
            result[0] += 0.0264029707592654;
          } else {
            result[0] += -0.00743452340283382;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += 0.015637148957569107;
          } else {
            result[0] += -0.06752858175980914;
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
          result[0] += -0.014857113791804029;
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.86220884323120206) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.566809177398682529) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.407877445220948154) ) ) {
                  result[0] += 0.0019989841297884713;
                } else {
                  result[0] += -0.03870811332477902;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                  result[0] += -0.011594583822211338;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                    result[0] += 0.035336607757661;
                  } else {
                    result[0] += -0.08287332841469834;
                  }
                }
              }
            } else {
              result[0] += 0.06441130454409015;
            }
          } else {
            if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              result[0] += -0.008453024977600185;
            } else {
              result[0] += 0.034006325210540436;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.018225350811205333;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.024152306834987397;
          } else {
            result[0] += -0.10996371316541922;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.680079460144043857) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.04989885135797073;
          } else {
            result[0] += -0.013237436732209222;
          }
        } else {
          result[0] += 0.007421071169702755;
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.676220536231995073) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
              result[0] += -0.01477953058106777;
            } else {
              result[0] += -0.0651444195757196;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)14.32165384292602717) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.80760431289673029) ) ) {
                result[0] += -0.07513757923848084;
              } else {
                result[0] += -0.12006466816425848;
              }
            } else {
              result[0] += 0.06641031262589052;
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7.83509850502014249) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
              result[0] += -0.016367197700610608;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.823264837265015537) ) ) {
                result[0] += 0.0010337481376999503;
              } else {
                result[0] += 0.04339704948577975;
              }
            }
          } else {
            result[0] += -0.19541467849462235;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      result[0] += 0.00042023892975793945;
    } else {
      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
          result[0] += -0.01003018229891671;
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.66339445114135831) ) ) {
                result[0] += 0.006664155501584256;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
                        result[0] += -0.005951219303215623;
                      } else {
                        if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                          result[0] += -0.059258378830047864;
                        } else {
                          result[0] += 0.05864537132972284;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                          result[0] += -0.003186396777158185;
                        } else {
                          result[0] += 0.016804423087561162;
                        }
                      } else {
                        result[0] += -0.011344962590247684;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06957215411011292;
                    } else {
                      result[0] += -0.012816548906364217;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.021413892638858137;
                  } else {
                    if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                        result[0] += 0.014618097111228115;
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.061140976655877755;
                        } else {
                          result[0] += -0.02989505682438088;
                        }
                      }
                    } else {
                      result[0] += -0.06212663410556135;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.044529251930130215;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.0010154258557852447;
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.009217198100586443;
              } else {
                result[0] += -0.0007240143469504034;
              }
            }
          }
        }
      } else {
        result[0] += 0.021746848565767318;
      }
    }
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.385823249816895419) ) ) {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += 0.0022967224276843968;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.166635274887085849) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.48802232742309748) ) ) {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.008777565566595784;
                        } else {
                          result[0] += 0.04207590647736161;
                        }
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.002650976181031162) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                            result[0] += 0.0011045095863778148;
                          } else {
                            result[0] += 0.025598848142677124;
                          }
                        } else {
                          result[0] += -0.023296953538994983;
                        }
                      }
                    } else {
                      result[0] += -0.01707359583428152;
                    }
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += -0.02489483093560337;
                    } else {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                          if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += 0.11437881076352104;
                            } else {
                              result[0] += -0.09008679260715259;
                            }
                          } else {
                            if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.413873195648194248) ) ) {
                              result[0] += -0.06997227333399829;
                            } else {
                              result[0] += 0.05575867820491407;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.007240838440550021;
                            } else {
                              result[0] += -0.09585834366158422;
                            }
                          } else {
                            if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59600305557251154) ) ) {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                                    result[0] += 0.004037929207737254;
                                  } else {
                                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.83236885070800959) ) ) {
                                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                                        result[0] += 0.08226771324115563;
                                      } else {
                                        result[0] += -0.045176700824604836;
                                      }
                                    } else {
                                      result[0] += 0.05072391415693959;
                                    }
                                  }
                                } else {
                                  result[0] += 0.022836926651604125;
                                }
                              } else {
                                result[0] += 0.03713208620114442;
                              }
                            } else {
                              result[0] += -0.0859281013119642;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.029315835627988514;
                      }
                    }
                  }
                } else {
                  result[0] += 0.08034343094312879;
                }
              } else {
                result[0] += -0.011415838577333581;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                result[0] += -0.0404697482106375;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.04457092285156428) ) ) {
                  result[0] += -0.02111346281268269;
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      result[0] += 0.1257819607817413;
                    } else {
                      result[0] += 0.020668564139042434;
                    }
                  } else {
                    result[0] += -0.015078537351519093;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
              result[0] += -0.016048142306422455;
            } else {
              result[0] += -0.0017994941183154829;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05539035797119318) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.039789007620473715;
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                      result[0] += -0.04681100109409936;
                    } else {
                      result[0] += 0.0799761254761575;
                    }
                  } else {
                    result[0] += -0.10606184243647743;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                  result[0] += -0.07396812169646687;
                } else {
                  result[0] += 0.04904988642404696;
                }
              }
            } else {
              result[0] += 0.007342669104074249;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += 0.0012938190050553041;
        } else {
          result[0] += -0.0451478274249727;
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.468516826629640448) ) ) {
        result[0] += 0.006807063643370081;
      } else {
        result[0] += 0.03971724312292247;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.00029787336342935425;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.385823249816895419) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
          result[0] += 0.0007092433778497343;
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                result[0] += 0.015475496597656603;
              } else {
                result[0] += -0.01616914140123235;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.047907682775895716;
              } else {
                if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                    result[0] += -0.05767780792338756;
                  } else {
                    result[0] += -0.017547328384082832;
                  }
                } else {
                  result[0] += -0.0058512287767928505;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                result[0] += -0.028992417828709136;
              } else {
                result[0] += 0.04988883417005193;
              }
            } else {
              result[0] += 0.003882716366823694;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
              result[0] += -0.015243171289662447;
            } else {
              result[0] += 0.0050539331840319945;
            }
          } else {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0026912553611649665;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                      result[0] += 0.057411616916505086;
                    } else {
                      result[0] += -0.004879686388227868;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.026556532208074152;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                              result[0] += -0.08740230711153402;
                            } else {
                              result[0] += 0.0826472541888551;
                            }
                          } else {
                            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.757321834564209873) ) ) {
                              result[0] += 0.009652716296415226;
                            } else {
                              result[0] += -0.024005190537460343;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.05033268276800639;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                        result[0] += -0.03264209908407275;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                          result[0] += -0.0007237090977632444;
                        } else {
                          result[0] += 0.03172718934882607;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                    result[0] += 0.008821692842476029;
                  } else {
                    result[0] += -0.06293649436679695;
                  }
                }
              } else {
                result[0] += 0.02915317078896599;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
              result[0] += 0.009142727153894768;
            } else {
              result[0] += 0.06221772051975061;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += 0.007009193034328065;
              } else {
                result[0] += -0.016577163497906253;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.05375322628524898;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                      result[0] += 0.0007585204454134789;
                    } else {
                      result[0] += -0.010112716348559384;
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.04558744783880102;
                      } else {
                        result[0] += 0.0031024537721017417;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.025917971151390717;
                        } else {
                          result[0] += -0.012130671730071088;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.17202329635620295) ) ) {
                            result[0] += -0.006445157334436642;
                          } else {
                            result[0] += 0.00832100765070058;
                          }
                        } else {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.010643566122030875;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += -0.0597280307870503;
                                } else {
                                  result[0] += 0.05101764145673539;
                                }
                              } else {
                                if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                                  result[0] += -0.08569818713043252;
                                } else {
                                  result[0] += 0.00547364158041088;
                                }
                              }
                            } else {
                              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                  result[0] += -0.037163842036103185;
                                } else {
                                  result[0] += 0.04468790233072466;
                                }
                              } else {
                                result[0] += 0.07677495916842936;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        result[0] += 0.01257824396583058;
                      } else {
                        result[0] += -0.11442416937263314;
                      }
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.04837839519871178;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                          result[0] += -0.09528351805656816;
                        } else {
                          result[0] += 0.01543224579989759;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += -0.004540470815906963;
                      } else {
                        result[0] += -0.08737706361638153;
                      }
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += -0.00887804854622821;
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.04891067796904112;
                        } else {
                          result[0] += 0.012670371746166137;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
                    result[0] += -0.0605491930169239;
                  } else {
                    result[0] += 0.005148895878870841;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.468516826629640448) ) ) {
        result[0] += 0.006324754843561455;
      } else {
        result[0] += 0.03822664383610729;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.000306648764747871;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.09753179550171076) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                  result[0] += -0.03954720550879579;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    result[0] += 0.0028245224935318975;
                  } else {
                    result[0] += 0.028001040721784484;
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01590308164997314;
                  } else {
                    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        result[0] += 0.00777971070020907;
                      } else {
                        result[0] += -0.019112471913520668;
                      }
                    } else {
                      result[0] += -0.00944095871215386;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.087577104568482333) ) ) {
                    result[0] += -0.029553733808336632;
                  } else {
                    result[0] += -0.0014384876195483354;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                  result[0] += -0.013425988499134593;
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.03025248691008449;
                    } else {
                      result[0] += -0.004143966667446821;
                    }
                  } else {
                    result[0] += -0.02936540398669046;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.012579621843196585;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                          result[0] += -0.0814647096234743;
                        } else {
                          result[0] += 0.012749217755418919;
                        }
                      } else {
                        result[0] += -0.021630697854371737;
                      }
                    }
                  } else {
                    result[0] += 0.007143822888857729;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
                          result[0] += 0.014527025796800208;
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.08463759417131567;
                          } else {
                            result[0] += -0.03103983373407461;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.015203161577866399;
                        } else {
                          result[0] += -0.007105626088070151;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.04083856436221703;
                      } else {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.023405816546420977;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
                            result[0] += -0.04731043588168578;
                          } else {
                            result[0] += 0.007569959441146074;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.01740073763107685;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                result[0] += -0.004242533367659875;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.050656174457310715;
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    result[0] += -0.04089754865485314;
                  } else {
                    result[0] += -0.012062220733748165;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                    result[0] += -0.029618051238233557;
                  } else {
                    result[0] += 0.04628577701949407;
                  }
                } else {
                  result[0] += 0.0050881201817556876;
                }
              } else {
                result[0] += -0.07534123386409877;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.0014474860322457326;
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                      result[0] += -0.040379023747795305;
                    } else {
                      if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.05838359472114701;
                      } else {
                        result[0] += 0.0028504571576604687;
                      }
                    }
                  } else {
                    result[0] += 0.03280157901517138;
                  }
                }
              } else {
                result[0] += 0.006526158202462125;
              }
            } else {
              result[0] += -0.056783784698830386;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.03943759265885967;
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.022466977813572847;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                    result[0] += 0.05508634939701476;
                  } else {
                    result[0] += 0.006153561181315773;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.06462960809617697;
              } else {
                result[0] += 0.022676921124464587;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.954540252685547763) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
              result[0] += -0.006541288958358879;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                result[0] += -0.00110965991667697;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
                  result[0] += -0.0039978869918029765;
                } else {
                  result[0] += 0.02302276888793295;
                }
              }
            }
          } else {
            result[0] += -0.01472397216595307;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                result[0] += 0.013979190416351093;
              } else {
                result[0] += -0.04248438669879956;
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                result[0] += -0.04220231982613399;
              } else {
                result[0] += -0.07774965077229677;
              }
            }
          } else {
            result[0] += 0.025327740134547617;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
          result[0] += -0.045615984124565113;
        } else {
          result[0] += 0.01704011973771758;
        }
      } else {
        result[0] += 0.002200393653197572;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.0003092445488851041;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.468516826629640448) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.257356405258179155) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.67080307006836115) ) ) {
                        result[0] += 0.02635182879458588;
                      } else {
                        result[0] += -0.022827551330746876;
                      }
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                            result[0] += 0.01761024836125016;
                          } else {
                            result[0] += -0.0007254395642567815;
                          }
                        } else {
                          result[0] += 0.03403221548244021;
                        }
                      } else {
                        result[0] += -0.009024080874326981;
                      }
                    }
                  } else {
                    result[0] += -0.00820812098738744;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                    result[0] += -0.02191093571295436;
                  } else {
                    result[0] += -0.00566436224244536;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                    result[0] += -0.012257843691329092;
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                      result[0] += 0.023950851392289058;
                    } else {
                      result[0] += -0.02602722612471502;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.015491354278279156;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                            result[0] += -0.07326928866007762;
                          } else {
                            result[0] += 0.011620410553543329;
                          }
                        } else {
                          result[0] += -0.0197546161689834;
                        }
                      }
                    } else {
                      result[0] += 0.006684581598099353;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += 0.009791204582902108;
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.54296922683715998) ) ) {
                            result[0] += 0.030907962551839564;
                          } else {
                            result[0] += -0.01093123358586634;
                          }
                        } else {
                          result[0] += -0.02824176541283351;
                        }
                      }
                    } else {
                      result[0] += -0.014919488872045902;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.572496652603150302) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.075335502624512607) ) ) {
                    result[0] += 0.0222030700074564;
                  } else {
                    result[0] += -0.006471260086313532;
                  }
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.007691251289938528;
                  } else {
                    result[0] += -0.0223660655314456;
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.99033999443054288) ) ) {
                    result[0] += 0.026225401799133338;
                  } else {
                    result[0] += -0.0012417895547498304;
                  }
                } else {
                  result[0] += 0.058387569296706236;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.032353814286623664;
                  } else {
                    result[0] += -0.0022258482323242513;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                      result[0] += 0.03937192906779145;
                    } else {
                      result[0] += 0.01044359336087542;
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.613121509552002841) ) ) {
                          result[0] += -0.010716289784077675;
                        } else {
                          result[0] += 0.00907817220553742;
                        }
                      } else {
                        result[0] += -0.036687395857856984;
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                        result[0] += -0.030070979901754332;
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += 0.014545435122937246;
                        } else {
                          result[0] += -0.02709806042888349;
                        }
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.05113303646357368;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.03564786971873082;
                } else {
                  result[0] += 0.005373908558245716;
                }
              } else {
                result[0] += 0.03549843797600275;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
            if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                result[0] += -0.0027630937827567295;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.014495701319741136;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                    result[0] += 0.011004201860951099;
                  } else {
                    result[0] += -0.0772908691880426;
                  }
                }
              }
            } else {
              result[0] += -0.008877394576564059;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.954540252685547763) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                result[0] += -0.007916784047593751;
              } else {
                result[0] += -0.04883256126225923;
              }
            } else {
              result[0] += 0.021551876647164445;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
            result[0] += -0.00010910957465513057;
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += -0.011491414904028545;
            } else {
              result[0] += -0.045063019671275316;
            }
          }
        } else {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += -0.004325148850086661;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.028173005127818375;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                  result[0] += -0.030738312029662148;
                } else {
                  result[0] += 0.02543745174174642;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.201166391372681552) ) ) {
                result[0] += 0.0023219843226827577;
              } else {
                result[0] += 0.01163636023139169;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
        result[0] += 0.01984768495324702;
      } else {
        result[0] += 0.15996061687498397;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.00031774953491419165;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.422362327575684482) ) ) {
      if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
              result[0] += 0.0004932080285684016;
            } else {
              result[0] += -0.036948212053421185;
            }
          } else {
            result[0] += -0.005535373539718214;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.791641235351563388) ) ) {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                    result[0] += -0.008733221660308507;
                  } else {
                    result[0] += -0.055106252379248905;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.249904870986938921) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.32868957519531428) ) ) {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.0982954835695301;
                        } else {
                          result[0] += 0.006299832658312002;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                          result[0] += 0.08643294905953401;
                        } else {
                          result[0] += 0.020915240200120463;
                        }
                      }
                    } else {
                      result[0] += -0.005779079332838988;
                    }
                  } else {
                    result[0] += 0.07748799589515688;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 0.007067536736546275;
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.03448712260188013;
                  } else {
                    result[0] += -0.0884436901919805;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.09136780537331352;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.308072090148926669) ) ) {
                      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.0338891195523325;
                      } else {
                        result[0] += 0.01036884688381176;
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.012438650394975935;
                      } else {
                        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += 0.02161107878377419;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.559038162231446201) ) ) {
                            if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                              result[0] += -0.01566327127654886;
                            } else {
                              result[0] += 0.017679602581613035;
                            }
                          } else {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.428993701934816229) ) ) {
                              result[0] += -0.07535834766768706;
                            } else {
                              result[0] += -0.005210066769239072;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.051500320434572089) ) ) {
                        result[0] += 8.039424067819779e-05;
                      } else {
                        result[0] += 0.032452486121079836;
                      }
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.845905780792238104) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.092434883117676669) ) ) {
                            result[0] += -0.10782589571583108;
                          } else {
                            result[0] += -0.009169852656882712;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                            result[0] += -0.06163592677950507;
                          } else {
                            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)15.24862766265869318) ) ) {
                              result[0] += 0.04803306758468499;
                            } else {
                              result[0] += 0.1644368351511443;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.01904082179864158;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += 0.03988245170222482;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0229236275830665;
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.004649271965550507;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        result[0] += 0.0387884331256904;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += -0.06634257789655891;
                        } else {
                          result[0] += -0.0070297609806820975;
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
              result[0] += 0.004351094853422871;
            } else {
              result[0] += -0.03600780921006732;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0021388658370468916;
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.0019488610515412464;
              } else {
                result[0] += -0.019115748398383595;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                result[0] += 0.03808099388755846;
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.182021141052246982) ) ) {
                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.772694945335388628) ) ) {
                      result[0] += -0.004787227680037585;
                    } else {
                      result[0] += -0.02337581559947048;
                    }
                  } else {
                    result[0] += 0.00458104728624261;
                  }
                } else {
                  result[0] += 0.013344856614358906;
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.01937151632383387;
                    } else {
                      result[0] += -0.04491923304742358;
                    }
                  } else {
                    result[0] += -0.05167675611148203;
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.07765625705754632;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                      result[0] += -0.03941911050123739;
                    } else {
                      result[0] += 0.04074236132246992;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.007242791758002487;
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.55507802963257014) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.021934409024249216;
                          } else {
                            result[0] += -0.02819049170398734;
                          }
                        } else {
                          result[0] += 0.02130320272039146;
                        }
                      } else {
                        result[0] += 0.05128928166849283;
                      }
                    }
                  } else {
                    result[0] += -0.1079042544767359;
                  }
                } else {
                  result[0] += 0.041800178176080734;
                }
              }
            }
          }
        } else {
          result[0] += -0.04515263216751846;
        }
      }
    } else {
      result[0] += 0.006161071007534514;
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    result[0] += 0.00028948574646147843;
  } else {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += -0.009067609406250563;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.08449630464520112;
                  } else {
                    result[0] += 0.026757534890472535;
                  }
                } else {
                  result[0] += 0.01370317781177326;
                }
              }
            } else {
              result[0] += 0.004094723761665202;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.745876312255860263) ) ) {
                result[0] += 0.006397331823555912;
              } else {
                result[0] += -0.041458763289900516;
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.07580690369568381;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += 0.03963052142966006;
                      } else {
                        result[0] += -0.03284057926690297;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                        result[0] += -0.02217290203518882;
                      } else {
                        result[0] += -0.06340392125761979;
                      }
                    } else {
                      result[0] += -0.004684349336486557;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    result[0] += 0.05043095268086737;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.44140100479126021) ) ) {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.003848585487276321;
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += -0.026230041105586594;
                        } else {
                          result[0] += 0.061005192311526214;
                        }
                      }
                    } else {
                      result[0] += -0.07702475848818466;
                    }
                  }
                }
              } else {
                result[0] += 0.013138443545621329;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.000308036804200107) ) ) {
                result[0] += -0.00019984575040986118;
              } else {
                result[0] += -0.029575609619715084;
              }
            } else {
              if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += -0.011860919322048176;
              } else {
                result[0] += -0.03321628399060388;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)6.500000000000000888) ) ) {
              result[0] += 0.004499919062648761;
            } else {
              result[0] += -0.017866088323797668;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81197786331176935) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.00595356264221743;
              } else {
                result[0] += 0.06491391366013081;
              }
            } else {
              result[0] += -0.01901158976319669;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.014552136708860406;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.198464870452881303) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.008298798898678522;
                } else {
                  result[0] += -0.027538395386349435;
                }
              } else {
                result[0] += 0.1122740325627303;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.06474608421017253;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.02497905559897267;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.03028712394514469;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.01157977620699397;
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                          result[0] += 0.04868514876653924;
                        } else {
                          result[0] += -0.03587169838935297;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.04942620921974555;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)6.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    result[0] += -0.07907023440462468;
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += -0.028769899911324073;
                    } else {
                      result[0] += 0.028980070189929666;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
                    result[0] += -0.0008386585708845079;
                  } else {
                    result[0] += 0.01569644761636124;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += -0.054398750568838194;
                    } else {
                      result[0] += 0.03207107280565788;
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.05727982783180851;
                    } else {
                      result[0] += 0.019670206779033163;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      result[0] += 0.01157212786300244;
                    } else {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                          result[0] += -0.08911285027881284;
                        } else {
                          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.04386219448063954;
                          } else {
                            result[0] += -0.025525771548277287;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.11069996524487674;
                        } else {
                          result[0] += -0.026115055685225833;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.014291968406623737;
                  }
                }
              }
            }
          } else {
            result[0] += -0.01428091251130394;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
          result[0] += -0.007365612281402437;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
            result[0] += -0.016788376175063978;
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.030690811703322413;
            } else {
              result[0] += -0.06968274256947715;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.278777599334717685) ) ) {
          result[0] += -0.00398373940983354;
        } else {
          result[0] += 0.1961210820130318;
        }
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.0003280215902195333;
  } else {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                result[0] += 0.006813139232260764;
              } else {
                result[0] += -0.06152476785904966;
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += -0.0028142005529806346;
                  } else {
                    result[0] += 0.11949938959615357;
                  }
                } else {
                  result[0] += -0.02052040069520431;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                  result[0] += -0.04000456190460926;
                } else {
                  result[0] += 0.024248855051677034;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += -0.02195955557656948;
              } else {
                result[0] += 0.03745264319775946;
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.772996187210083896) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)7.278777599334717685) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.234983921051027167) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.0133657734577943;
                            } else {
                              result[0] += 0.04415965070378122;
                            }
                          } else {
                            result[0] += -0.015027746534400166;
                          }
                        } else {
                          result[0] += 0.06729571933574038;
                        }
                      } else {
                        result[0] += -0.026743864429334026;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.13183502059431482;
                      } else {
                        result[0] += 0.02860084429185269;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                        result[0] += -0.015627158534644216;
                      } else {
                        result[0] += 0.001702368455359864;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
                          result[0] += 0.036550537789197916;
                        } else {
                          result[0] += -0.04564351988961889;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.925687789916993964) ) ) {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                            result[0] += 0.006427332528912271;
                          } else {
                            result[0] += -0.08530585937890656;
                          }
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.08843347570940932;
                          } else {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.734558105468750888) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59600305557251154) ) ) {
                                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                                  result[0] += 0.0012582243822875175;
                                } else {
                                  result[0] += -0.05075988503574844;
                                }
                              } else {
                                result[0] += 0.03922729584665811;
                              }
                            } else {
                              result[0] += -0.09696599066119431;
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.10835689231842094;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.010498382426505243;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
                        result[0] += -0.003704631821820882;
                      } else {
                        result[0] += -0.18395892677450268;
                      }
                    } else {
                      result[0] += 0.03619818350365179;
                    }
                  }
                } else {
                  result[0] += 0.019155278142094428;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.037894570006517164;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                result[0] += 0.012646710880314075;
              } else {
                result[0] += -0.006029262564153848;
              }
            }
          } else {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.138696432113648349) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.00047005502438947247;
                } else {
                  result[0] += -0.045382859066332215;
                }
              } else {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.10718667539788329;
                  } else {
                    result[0] += -0.0037960162115469316;
                  }
                } else {
                  result[0] += -0.0739522422555727;
                }
              }
            } else {
              result[0] += -0.05700576344519708;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.06868470229861288;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.09244903823567319;
            } else {
              result[0] += -0.01374612509370295;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.01670693347074884;
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.017886140199407933;
                        } else {
                          result[0] += -0.01735216990497038;
                        }
                      } else {
                        result[0] += -0.05372742496239043;
                      }
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.169590950012207919) ) ) {
                          result[0] += 0.011908562664133857;
                        } else {
                          result[0] += 0.0620004772202648;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.20279194034344405;
                          } else {
                            result[0] += -0.0354597625843672;
                          }
                        } else {
                          result[0] += -0.03463930018462805;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.026646672111502054;
                  }
                } else {
                  result[0] += 0.004238011553860147;
                }
              }
            } else {
              result[0] += -0.013208622570851201;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.497866153717041238) ) ) {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
            result[0] += 0.0017034290996629621;
          } else {
            result[0] += 0.013721653944903734;
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.0071319869247845405;
          } else {
            result[0] += -0.036612969617498214;
          }
        }
      } else {
        result[0] += -0.04306112023214148;
      }
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.00030803110569870175;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)5.500000000000000888) ) ) {
                result[0] += -0.01361504503315929;
              } else {
                result[0] += -0.09918780530365254;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.06729796135425267;
              } else {
                result[0] += -0.045940041563381086;
              }
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.04227644312431553;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                result[0] += 0.0539174673324441;
              } else {
                result[0] += 0.0008503550050344092;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.016423660961643445;
                  } else {
                    result[0] += -0.029459078757137894;
                  }
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.0649209560050758;
                  } else {
                    result[0] += 0.049201880935545145;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                  result[0] += -0.04056095102311591;
                } else {
                  result[0] += 0.047245134477892047;
                }
              }
            } else {
              if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += 0.022146311979445766;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.568724632263184482) ) ) {
                      result[0] += 0.013704720100253405;
                    } else {
                      result[0] += -0.04035442097207728;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.169590950012207919) ) ) {
                    result[0] += -0.00045711807247643984;
                  } else {
                    result[0] += -0.09261954181221531;
                  }
                }
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.109050035476685458) ) ) {
                    result[0] += 7.408358673046463e-05;
                  } else {
                    result[0] += -0.23490274453562515;
                  }
                } else {
                  result[0] += 0.02184093318315336;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                result[0] += 0.033328967506083045;
              } else {
                result[0] += -0.006097687058263437;
              }
            } else {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.046238811417863776;
                } else {
                  result[0] += 0.0019216173259077461;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.025034087529870436;
                  } else {
                    result[0] += 0.04240594174629621;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                    result[0] += 0.004690522970746139;
                  } else {
                    result[0] += 0.10139152210962046;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.809154510498047763) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 0.0009341515983336103;
          } else {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += -0.00019707641524151716;
            } else {
              result[0] += -0.00570296480172422;
            }
          }
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
            result[0] += 0.0065244282814202375;
          } else {
            result[0] += 0.04118570820416901;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
          result[0] += 0.004126636865524217;
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.556798219680787021) ) ) {
            result[0] += 0.0007938185387819975;
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)9.833202838897706854) ) ) {
              result[0] += -0.03781492719927684;
            } else {
              result[0] += 0.12568794569593414;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.688566803932190385) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
              result[0] += 0.024469384292779298;
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                result[0] += 0.051753831898317795;
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08835037570795318;
                    } else {
                      result[0] += -0.010791913347868887;
                    }
                  } else {
                    result[0] += 0.008016599671112718;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.243503808975220615) ) ) {
                    result[0] += -0.01000871455728777;
                  } else {
                    result[0] += -0.04200598584217222;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.549068689346314365) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.09844876640863659;
                } else {
                  result[0] += 0.0033475957701924633;
                }
              } else {
                if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.632926940917970526) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                          result[0] += -0.015548475651983432;
                        } else {
                          result[0] += 0.03878841426327254;
                        }
                      } else {
                        result[0] += -0.05208492208725496;
                      }
                    } else {
                      result[0] += 0.04489578523648598;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                      result[0] += 0.04117749109775017;
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.03934340298286271;
                      } else {
                        result[0] += 0.007124675937363001;
                      }
                    }
                  }
                } else {
                  result[0] += 0.004965056901092987;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.035443881071232045;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                    result[0] += 0.028774706534403588;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.623641014099121982) ) ) {
                        result[0] += 1.3266306857861262e-05;
                      } else {
                        result[0] += -0.07838287903115133;
                      }
                    } else {
                      result[0] += 0.01498663756654716;
                    }
                  }
                }
              } else {
                result[0] += 0.04687036934363176;
              }
            }
          }
        } else {
          result[0] += -0.03991310125159497;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.00019943854023705444;
      } else {
        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
          result[0] += -0.0007726512950037554;
        } else {
          result[0] += -0.009115472115437742;
        }
      }
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.556798219680787021) ) ) {
                        result[0] += 0.006096829737078097;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                          result[0] += 0.014880759414164888;
                        } else {
                          result[0] += -0.040497171652290456;
                        }
                      }
                    } else {
                      result[0] += -0.05821074649187427;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                      result[0] += 0.01307365530737561;
                    } else {
                      result[0] += -0.00865150412190858;
                    }
                  }
                } else {
                  result[0] += -0.11736852117274921;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.422362327575684482) ) ) {
                  result[0] += -0.0014087323431223294;
                } else {
                  result[0] += 0.03128516078352975;
                }
              }
            } else {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                result[0] += 0.05130707763899761;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.749434947967529741) ) ) {
                  result[0] += 0.01923856811030894;
                } else {
                  result[0] += -0.036244764560001;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += 0.13088451181479935;
                } else {
                  result[0] += -0.08306562996217937;
                }
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.274755001068116123) ) ) {
                    result[0] += 0.004257664242742878;
                  } else {
                    result[0] += -0.04323968248558388;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.305786132812500888) ) ) {
                    result[0] += -0.019793500918922815;
                  } else {
                    result[0] += 0.05027710381639721;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                  result[0] += -0.008169289438790915;
                } else {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                      result[0] += 0.015876106552185548;
                    } else {
                      if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.009363549070569235;
                      } else {
                        result[0] += 0.02072011160074851;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.048207110227422315;
                    } else {
                      result[0] += -0.0042065880039567445;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += 0.028012708835994854;
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.645421981811524326) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        result[0] += 0.014285846999412783;
                      } else {
                        result[0] += -0.011084633749244694;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.860674262046814409) ) ) {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.004872741836061849;
                            } else {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                                result[0] += 0.0620877213447699;
                              } else {
                                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95751476287841975) ) ) {
                                  result[0] += -0.14112812317946755;
                                } else {
                                  result[0] += 0.07508580606866876;
                                }
                              }
                            }
                          } else {
                            result[0] += -0.029516861436411808;
                          }
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                              result[0] += -0.04446202139742335;
                            } else {
                              result[0] += 0.015017384425554649;
                            }
                          } else {
                            result[0] += 0.006094312359871266;
                          }
                        }
                      } else {
                        result[0] += -0.0242120983599807;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += -0.020132962674169766;
                    } else {
                      result[0] += 0.010372530325399733;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                          result[0] += 0.016695461423303488;
                        } else {
                          result[0] += -0.0324771761840795;
                        }
                      } else {
                        result[0] += 0.02084597858724079;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81839752197265803) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.382196187973023349) ) ) {
                          result[0] += -0.018129309614881843;
                        } else {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                            result[0] += 0.03623764239919021;
                          } else {
                            result[0] += -0.026055447098027176;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                          result[0] += 0.09150906166661764;
                        } else {
                          result[0] += 0.03305929783665802;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)5.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.10091206647599182;
              } else {
                result[0] += -0.07684197975116931;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11581277847290217) ) ) {
                result[0] += 0.006626127191334707;
              } else {
                result[0] += -0.053746386879493895;
              }
            }
          } else {
            result[0] += 0.058102365425260816;
          }
        }
      } else {
        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.469231128692627841) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
              result[0] += 0.03418110043738198;
            } else {
              result[0] += -0.054176488797841385;
            }
          } else {
            result[0] += -0.07382366035585906;
          }
        } else {
          result[0] += 0.0029679468841414714;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += -0.00042394305968426684;
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
          result[0] += -0.013179630044880382;
        } else {
          if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            result[0] += 0.014638043643176963;
          } else {
            result[0] += -0.05276686400474965;
          }
        }
      } else {
        result[0] += -0.004368659664943715;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
        result[0] += -0.00022498674913071128;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.012684366651771953;
                      } else {
                        result[0] += -0.050559596439664525;
                      }
                    } else {
                      result[0] += -0.001832737397461366;
                    }
                  } else {
                    result[0] += -0.11736205893315002;
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.422362327575684482) ) ) {
                      result[0] += -0.004106978347237757;
                    } else {
                      result[0] += 0.02882227455575396;
                    }
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0010189243870210214;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                        result[0] += 0.015892523507365502;
                      } else {
                        result[0] += 0.08981046987049625;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.051311465593445904;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.10538721084594904) ) ) {
                    result[0] += 0.02455608028047721;
                  } else {
                    result[0] += -0.02973983325294681;
                  }
                }
              }
            } else {
              result[0] += 0.0025600824600173007;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += 0.02558695687516818;
              } else {
                result[0] += -0.06281549513799665;
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.12961053500780464;
              } else {
                result[0] += 0.013663805176818667;
              }
            }
          }
        } else {
          result[0] += 0.0025944333187049307;
        }
      }
    } else {
      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.137252807617188388) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  result[0] += 0.02814524382315177;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
                      result[0] += 0.011967830908583521;
                    } else {
                      result[0] += -0.15770389469940835;
                    }
                  } else {
                    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.009987381671044534;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                        result[0] += 0.059360201472757994;
                      } else {
                        result[0] += -0.002634362931825687;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                  result[0] += -0.0020937743680177236;
                } else {
                  result[0] += -0.0614028194448347;
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.00139282449080424;
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.052366961225696444;
                } else {
                  result[0] += 0.004939066745934983;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.004817990045938988;
                  } else {
                    result[0] += 0.007557427380555447;
                  }
                } else {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.00883341960827293;
                          } else {
                            result[0] += -0.12006617959091666;
                          }
                        } else {
                          result[0] += 0.07611341350466576;
                        }
                      } else {
                        if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.011109445265609703;
                        } else {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.645421981811524326) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                              result[0] += -0.023348060080233587;
                            } else {
                              result[0] += -0.1366413329876926;
                            }
                          } else {
                            result[0] += 0.009329768441657692;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.08003593229488425;
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.06922666709329552;
                    } else {
                      result[0] += -0.015404263479037167;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)8.360008716583253729) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.005174303370491271;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        result[0] += 0.04484815137095642;
                      } else {
                        result[0] += -0.026064003039244273;
                      }
                    }
                  } else {
                    result[0] += -0.13148166289374638;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                    result[0] += 0.001612170041528904;
                  } else {
                    result[0] += 0.022575350250162718;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.001226087989168174;
              } else {
                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.020198368582917115;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
                    result[0] += -0.04224276451917154;
                  } else {
                    result[0] += 0.009324595351971658;
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.015035375457531131;
              } else {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                        result[0] += 0.015310485145306205;
                      } else {
                        result[0] += -0.13515818726538434;
                      }
                    } else {
                      result[0] += 0.10696902600592162;
                    }
                  } else {
                    result[0] += -0.023852963316050895;
                  }
                } else {
                  result[0] += 0.023464905660462013;
                }
              }
            } else {
              result[0] += 0.012016931332897354;
            }
          } else {
            result[0] += -0.0009564307567653417;
          }
        }
      } else {
        result[0] += -0.013616565748120594;
      }
    }
  } else {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
      result[0] += -0.032483440560368136;
    } else {
      result[0] += 0.01001578032128385;
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
    result[0] += 0.0002725136276973706;
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.008431667361501258;
                } else {
                  result[0] += 0.06224981959458044;
                }
              } else {
                result[0] += 0.08162241028322126;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.84983205795288264) ) ) {
                result[0] += 0.01905297957632529;
              } else {
                result[0] += -0.049860627937941465;
              }
            }
          } else {
            result[0] += 0.005007467449284185;
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += 0.004272256027135889;
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
              result[0] += -0.04150053072377291;
            } else {
              result[0] += 0.16750990939778698;
            }
          }
        }
      } else {
        result[0] += 0.000232330918787448;
      }
    } else {
      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.511434078216553178) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
              result[0] += -0.03029748980722658;
            } else {
              result[0] += 0.06749264419774086;
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.158124685287476474) ) ) {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                      result[0] += -0.008699895606920837;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                              result[0] += -0.030204090228625066;
                            } else {
                              result[0] += 0.11070140519654664;
                            }
                          } else {
                            result[0] += -0.011908827806272353;
                          }
                        } else {
                          result[0] += 0.009336540278309656;
                        }
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.0009821901964165614;
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.586156606674195224) ) ) {
                                result[0] += -0.002089288377689032;
                              } else {
                                result[0] += 0.03953457507886211;
                              }
                            }
                          } else {
                            result[0] += 0.04419310098356419;
                          }
                        } else {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.158509254455567294) ) ) {
                                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                                  result[0] += -0.07671383382220975;
                                } else {
                                  result[0] += 0.010798642559113203;
                                }
                              } else {
                                result[0] += 0.040220864108172175;
                              }
                            } else {
                              result[0] += 0.07555473721418574;
                            }
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.08721123494983062;
                            } else {
                              result[0] += 0.0001371507879019827;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0071683180747062845;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                    if ( UNLIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.06621233037131659;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.030981287866232138;
                      } else {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.06927398596803878;
                        } else {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.15100884437561124) ) ) {
                            result[0] += 0.12002467130386836;
                          } else {
                            result[0] += -0.03475017196327336;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
                      result[0] += -0.04609216136895346;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                        result[0] += 0.037501082709424524;
                      } else {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += -0.042357315821253856;
                          } else {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                              result[0] += -0.05089597547234919;
                            } else {
                              result[0] += 0.02723783356012709;
                            }
                          }
                        } else {
                          result[0] += -0.0036960841981498412;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.665046453475953037) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    result[0] += -0.03705555363708587;
                  } else {
                    result[0] += -0.0031267785414414787;
                  }
                } else {
                  result[0] += -0.0006215533767416637;
                }
              }
            } else {
              result[0] += -0.011126080530859303;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.807895898818970615) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.009326919685075106;
            } else {
              result[0] += -0.042285126854791535;
            }
          } else {
            result[0] += -0.03583404626723726;
          }
        }
      } else {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.021011038391767486;
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                result[0] += 0.002296426203252565;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.07138067851861339;
                } else {
                  result[0] += 0.004095962043688227;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.013468059320319687;
              } else {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                  result[0] += -0.02681296322054609;
                } else {
                  result[0] += -0.08618386615119493;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.650925397872925693) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.016258025640249068;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.053540229797365058) ) ) {
                    result[0] += -0.011773143958441985;
                  } else {
                    result[0] += 0.05254219201834074;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
                    result[0] += 0.031595852638867536;
                  } else {
                    result[0] += 0.0006823381222203375;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
                  result[0] += 0.020687170087425516;
                } else {
                  result[0] += 0.004928779214437672;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.036931514739991123) ) ) {
                  result[0] += -0.0617748280614059;
                } else {
                  result[0] += -0.0040116797577543755;
                }
              }
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)8.536084651947023261) ) ) {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += 9.981982225291632e-06;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.422362327575684482) ) ) {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += 0.018513921434336066;
                      } else {
                        result[0] += 0.06703333293571928;
                      }
                    } else {
                      result[0] += 0.052596857949698365;
                    }
                  } else {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.02732129505251592;
                    } else {
                      result[0] += -0.09250485008187873;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
                    result[0] += -0.1621409168786884;
                  } else {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.1325235210606826;
                    } else {
                      result[0] += 0.014300023376577937;
                    }
                  }
                }
              } else {
                result[0] += -0.014668154647479484;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                result[0] += -0.025491211754505357;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += 0.01588397613111326;
                  } else {
                    result[0] += -0.021800928061222257;
                  }
                } else {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.013985627250482463;
                    } else {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.883387088775636542) ) ) {
                          result[0] += 0.041913066555567334;
                        } else {
                          result[0] += -0.017379526744138358;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                          result[0] += -0.04792877951287389;
                        } else {
                          result[0] += -0.0052381571815019404;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                      result[0] += 0.03824219894537863;
                    } else {
                      result[0] += -0.048852947180847615;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.772694945335388628) ) ) {
              result[0] += 0.024018282507575525;
            } else {
              result[0] += 0.0628852744644173;
            }
          }
        } else {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.0016090006434766203;
          } else {
            result[0] += 0.05643745734356877;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            result[0] += 0.0024677521167467877;
          } else {
            result[0] += -0.026594297079974856;
          }
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.0024998459708867296;
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.378218650817871982) ) ) {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.030146097114007505;
                  } else {
                    result[0] += 0.013527103018990287;
                  }
                } else {
                  result[0] += 0.007227597625228029;
                }
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                  result[0] += -0.029529022872384483;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                        result[0] += -0.1743501628773457;
                      } else {
                        result[0] += -0.03684572210495029;
                      }
                    } else {
                      result[0] += 0.005260443559263094;
                    }
                  } else {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += 0.0058120358406461075;
                      } else {
                        result[0] += 0.05226956975642482;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                        result[0] += 0.02013939146016473;
                      } else {
                        result[0] += -0.007843002596835479;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.028540122239733463;
              }
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                    result[0] += 0.017334408385930332;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                      result[0] += 0.06683245645611774;
                    } else {
                      result[0] += -0.08284295786037534;
                    }
                  }
                } else {
                  result[0] += 0.006672913243580664;
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.601370334625245029) ) ) {
                      result[0] += -0.03115435545433172;
                    } else {
                      result[0] += -0.10006464583627535;
                    }
                  } else {
                    result[0] += -0.10196307947423063;
                  }
                } else {
                  result[0] += 0.0005985468102248095;
                }
              }
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                  result[0] += -0.007873074890976349;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.966960191726685458) ) ) {
                    result[0] += 0.012627511743719719;
                  } else {
                    result[0] += 0.05888378359967888;
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.06859588623047053) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07153234037216298;
                  } else {
                    result[0] += -0.014277579178623438;
                  }
                } else {
                  result[0] += 0.11745309115469255;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.014823999542716126;
              } else {
                result[0] += -0.07351293973736925;
              }
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.10099834330973437;
              } else {
                result[0] += -0.012344826073074983;
              }
            }
          } else {
            result[0] += -0.10817455798340847;
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.07061352847143729;
            } else {
              result[0] += -0.004126732168385874;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)14.32165384292602717) ) ) {
              result[0] += -0.07329297894726416;
            } else {
              result[0] += 0.0427947870325271;
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.08914141427586773;
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)8.536084651947023261) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
            result[0] += 0.006989562013049637;
          } else {
            result[0] += -0.0706685753696854;
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
            result[0] += -0.017225006920821677;
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
              result[0] += -0.04739319829180703;
            } else {
              result[0] += 0.09205876757277265;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.231051445007325107) ) ) {
              result[0] += -0.2875990176363054;
            } else {
              result[0] += 0.09391451944241179;
            }
          } else {
            result[0] += 0.11052534674316959;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.06298904646353565;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.008624163626920726;
              } else {
                result[0] += -0.0655772829080946;
              }
            } else {
              result[0] += 0.038708777319374305;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.00023576718986765413;
      } else {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.844439744949341042) ) ) {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.1479225158691424) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                    result[0] += -0.10475604602507181;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                      result[0] += -0.07750335548525258;
                    } else {
                      result[0] += 0.0026392790047089706;
                    }
                  }
                } else {
                  result[0] += 0.025740033861796754;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                  result[0] += 0.03678591174660688;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.43749904632568537) ) ) {
                    result[0] += -0.04134896295147514;
                  } else {
                    result[0] += -0.09642094586447923;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        result[0] += -0.06374454430803692;
                      } else {
                        result[0] += 0.005987549343413798;
                      }
                    } else {
                      result[0] += -0.050693556073796144;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
                      result[0] += -0.008380653240063247;
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                              result[0] += 0.04767250459452641;
                            } else {
                              result[0] += -0.03318610020044623;
                            }
                          } else {
                            result[0] += 0.00866211433550342;
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                            result[0] += -0.028068526590863715;
                          } else {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                                result[0] += -0.02382776034204072;
                              } else {
                                result[0] += 0.03478393000925335;
                              }
                            } else {
                              result[0] += 0.04663892428323263;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.519456863403321201) ) ) {
                          result[0] += 0.0023413235482268487;
                        } else {
                          result[0] += -0.044026352700453186;
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                      result[0] += -0.017234685550134823;
                    } else {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.802100181579590732) ) ) {
                          result[0] += -0.0001124912838315798;
                        } else {
                          result[0] += -0.06618849378387363;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                          result[0] += -0.0387967050087255;
                        } else {
                          result[0] += 0.017172070759394562;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                      result[0] += -0.03576075277647332;
                    } else {
                      result[0] += -0.009502557711708914;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                  result[0] += -0.06384980866433136;
                } else {
                  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.556798219680787021) ) ) {
                      result[0] += -0.020545371279182906;
                    } else {
                      result[0] += 0.025627984392327136;
                    }
                  } else {
                    result[0] += -0.0532688255865462;
                  }
                }
              }
            }
          } else {
            result[0] += -0.085239194430173;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      result[0] += 0.0526826494058149;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                        result[0] += 0.04423828602489488;
                      } else {
                        result[0] += 0.005136835559992364;
                      }
                    }
                  } else {
                    result[0] += 0.0028435719965989943;
                  }
                } else {
                  result[0] += -0.014872499530778793;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.035778118144136034;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)9.167253971099855292) ) ) {
                      result[0] += -0.00425390527590165;
                    } else {
                      result[0] += 0.1114399038298458;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0012325311568882852;
                  } else {
                    result[0] += -0.02723621859595403;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.012216310289405449;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                  result[0] += -0.0012740279496617957;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.05591974610526512;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.901921629905701128) ) ) {
                        result[0] += 0.018316880807636212;
                      } else {
                        result[0] += -0.007487182736027748;
                      }
                    }
                  } else {
                    result[0] += 0.003328912483458582;
                  }
                }
              }
            }
          } else {
            result[0] += -0.05916732301559027;
          }
        }
      }
    }
  } else {
    result[0] += -0.08914141427586773;
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)8.536084651947023261) ) ) {
    if ( UNLIKELY(  (data[27].missing != -1) && (data[27].fvalue <= (double)-1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.985194206237793857) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.00211572647094904) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
                result[0] += 0.053840428965297255;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                  result[0] += 0.03462647794391482;
                } else {
                  result[0] += -0.09876879892243852;
                }
              }
            } else {
              result[0] += -0.047147621875630714;
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
              result[0] += 0.009118673757320164;
            } else {
              result[0] += 0.13593991295665458;
            }
          }
        } else {
          result[0] += -0.06765930848893549;
        }
      } else {
        if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.231051445007325107) ) ) {
              result[0] += -0.19475160094027108;
            } else {
              result[0] += 0.09113023948934297;
            }
          } else {
            result[0] += 0.10857455498857474;
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += -0.07472969119574452;
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.008502819419009313;
              } else {
                result[0] += -0.06676651806098417;
              }
            } else {
              result[0] += 0.037786083902967976;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += 0.0017142362365283048;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
              result[0] += -0.010681126519624481;
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05539035797119318) ) ) {
                  result[0] += -0.0038506202533942283;
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.03757449072003057;
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.004064893365783503;
                    } else {
                      result[0] += 0.0236082756692821;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                  result[0] += -0.005938490654701473;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                    result[0] += -0.010680417920149106;
                  } else {
                    result[0] += 0.011623690015227273;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
              result[0] += 0.0003262348916215159;
            } else {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                      result[0] += -0.08507492358807861;
                    } else {
                      result[0] += 0.005010662311689414;
                    }
                  } else {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.556798219680787021) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.042063360764745;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
                            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.855921268463135654) ) ) {
                                result[0] += 0.051013848441155064;
                              } else {
                                result[0] += 0.1867003175059559;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                                result[0] += -0.07708437149758571;
                              } else {
                                result[0] += 0.04904614089402131;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
                                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                                  result[0] += 0.07359519989621194;
                                } else {
                                  result[0] += 2.2128391914019564e-05;
                                }
                              } else {
                                result[0] += 0.29099594051229255;
                              }
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.825982809066773349) ) ) {
                                  result[0] += 0.009489875562757108;
                                } else {
                                  result[0] += -0.028973414286466756;
                                }
                              } else {
                                result[0] += -0.03463933238732408;
                              }
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                            result[0] += 0.0028490134193169155;
                          } else {
                            result[0] += -0.020494103125118526;
                          }
                        } else {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.022381819866307797;
                          } else {
                            result[0] += -0.05659995139522653;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.003359783294836929;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                    result[0] += 0.008182586725581413;
                  } else {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.08942140340143061;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
                            result[0] += -0.036781250254943824;
                          } else {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.07598968589596329;
                            } else {
                              result[0] += -0.04987069442244947;
                            }
                          }
                        } else {
                          result[0] += 0.09044064308564434;
                        }
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += -0.0068598405851273084;
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                            result[0] += -0.03861828024001553;
                          } else {
                            result[0] += 0.018689936710941095;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.006299024861500985;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                    result[0] += -0.012817740511142942;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                      result[0] += -0.02333451500955009;
                    } else {
                      result[0] += -0.0006800634276065135;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.9054608345031756) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
                  result[0] += -0.0060703504882908205;
                } else {
                  result[0] += 0.008170370881099367;
                }
              } else {
                result[0] += -0.004059516136277099;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7.87937188148498624) ) ) {
                result[0] += -0.011830374777038535;
              } else {
                result[0] += 0.13880534030635436;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.994427680969239169) ) ) {
          result[0] += 0.0004365215687860841;
        } else {
          result[0] += 0.012267934854746688;
        }
      }
    }
  } else {
    result[0] += -0.08914141427586773;
  }
  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)8.536084651947023261) ) ) {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
      result[0] += 0.0002276438227967639;
    } else {
      if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.4822273254394549) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.14982107451405;
              } else {
                result[0] += -0.015422012392819002;
              }
            } else {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                result[0] += 0.07415319568934031;
              } else {
                result[0] += -0.05497208975289726;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                result[0] += 0.03254710191841439;
              } else {
                result[0] += -0.005994309215232588;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += 0.0533983072805829;
                  } else {
                    result[0] += -0.00620772972476824;
                  }
                } else {
                  result[0] += -0.14569203469777012;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.257122993469240058) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                              result[0] += -0.07138537755083223;
                            } else {
                              result[0] += 0.02760579003992228;
                            }
                          } else {
                            result[0] += 0.05900203386750402;
                          }
                        } else {
                          result[0] += -0.030752561783664024;
                        }
                      } else {
                        result[0] += -0.026468408894499862;
                      }
                    } else {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.807895898818970615) ) ) {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                          result[0] += 0.010757507001670698;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += -0.027552313557491295;
                            } else {
                              result[0] += -0.002273299834738143;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                                result[0] += 0.061080166764148514;
                              } else {
                                result[0] += -0.01406994363516461;
                              }
                            } else {
                              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += 0.09622930227452119;
                              } else {
                                result[0] += 0.02478599987190339;
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.0021814150861047153;
                        } else {
                          result[0] += -0.03224037768981852;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.700598716735840066) ) ) {
                      result[0] += 0.06810992653927876;
                    } else {
                      result[0] += 0.005259480502279137;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.033505359745640825;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                      result[0] += -0.011292547372753241;
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                        result[0] += -0.03776857126475214;
                      } else {
                        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                          result[0] += -0.04251916094753073;
                        } else {
                          result[0] += 0.0400191033504001;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)15.67977666854858576) ) ) {
                result[0] += 0.0334751448257918;
              } else {
                result[0] += -0.030092366438273472;
              }
            } else {
              result[0] += -0.030448546739346417;
            }
          } else {
            result[0] += -0.0015282277128628309;
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
            if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.47860431671142756) ) ) {
                result[0] += -0.03409552415466343;
              } else {
                result[0] += 0.02097508099643347;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                result[0] += 0.031860920185236945;
              } else {
                result[0] += -0.016995603831718185;
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += 0.010553751196515674;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
                    result[0] += 0.05025898354189458;
                  } else {
                    result[0] += -0.030470740486828924;
                  }
                }
              } else {
                result[0] += -0.03452915739464778;
              }
            } else {
              result[0] += 0.005952086503616869;
            }
          }
        } else {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.537586927413940874) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.10532989584151378;
              } else {
                result[0] += -0.0023359102468736895;
              }
            } else {
              result[0] += 0.009000591218862066;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.03421941000791247;
                  } else {
                    result[0] += -0.010665970413874312;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                      result[0] += -0.016483796447014133;
                    } else {
                      result[0] += -0.0558341199845273;
                    }
                  } else {
                    if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                            result[0] += 0.008032851683191974;
                          } else {
                            result[0] += -0.01477197782324235;
                          }
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.001892220396045784;
                          } else {
                            result[0] += 0.047085909543234855;
                          }
                        }
                      } else {
                        result[0] += -0.0341446993138439;
                      }
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.502698898315430576) ) ) {
                        result[0] += -0.029147195479512444;
                      } else {
                        result[0] += -0.15969489236639164;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.026162746324766135;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.417592287063599077) ) ) {
                  result[0] += -0.0600656200515634;
                } else {
                  result[0] += 0.006170991518153025;
                }
              } else {
                result[0] += 0.0067920923782850825;
              }
            }
          }
        }
      }
    }
  } else {
    result[0] += -0.08914141427586773;
  }
  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)12.00000000000000178) ) ) {
    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.729812622070313388) ) ) {
          if ( LIKELY(  (data[42].missing != -1) && (data[42].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.988812565803528276) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.802901029586792436) ) ) {
                        result[0] += -0.0066416868997498935;
                      } else {
                        result[0] += 0.04501258322978546;
                      }
                    } else {
                      result[0] += -0.019398370985966265;
                    }
                  } else {
                    result[0] += 0.00321217080480051;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
                    result[0] += 0.0003781753478643531;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += -0.022043915782120612;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += 0.033587133562415945;
                        } else {
                          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                            result[0] += 0.04465439217769266;
                          } else {
                            result[0] += -0.005602187487146439;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.011797620924541764;
                    }
                  }
                }
              } else {
                result[0] += 0.003356781096162582;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.66987490653991788) ) ) {
                result[0] += -0.015349161619828012;
              } else {
                result[0] += 0.09719892373233746;
              }
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.78399753570556818) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.448852539062500444) ) ) {
                      result[0] += -0.06323621376380938;
                    } else {
                      result[0] += 0.15641156558734;
                    }
                  } else {
                    result[0] += 0.019703568611731517;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                      result[0] += -0.05713062740674965;
                    } else {
                      result[0] += 0.09936236068603;
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.791486978530884677) ) ) {
                      result[0] += -0.08460229942578112;
                    } else {
                      result[0] += 0.07686394926481045;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.553655147552491123) ) ) {
                          result[0] += -0.02224491120034345;
                        } else {
                          result[0] += 0.06487898042826513;
                        }
                      } else {
                        result[0] += 0.021016472369437548;
                      }
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.04875459575179025;
                      } else {
                        result[0] += -0.008818448545033166;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.149475097656251776) ) ) {
                          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.11875578884504245;
                          } else {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                              result[0] += 0.005874249999385586;
                            } else {
                              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                                result[0] += -0.0670767493718214;
                              } else {
                                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                                  result[0] += -0.05367441128952021;
                                } else {
                                  result[0] += -0.0057880989124155105;
                                }
                              }
                            }
                          }
                        } else {
                          result[0] += 0.010705285045411104;
                        }
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.708768606185913974) ) ) {
                          result[0] += 0.003972891256211335;
                        } else {
                          result[0] += -0.018120970403805343;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                          result[0] += -0.009990454261285663;
                        } else {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.99098253250122248) ) ) {
                              result[0] += -0.00589541597757844;
                            } else {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.913499355316162998) ) ) {
                                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                                  result[0] += 0.11904262860668352;
                                } else {
                                  result[0] += 0.031965830270804456;
                                }
                              } else {
                                result[0] += -0.029026185827923035;
                              }
                            }
                          } else {
                            result[0] += 0.023700941174359183;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += -0.052533743769322294;
                        } else {
                          result[0] += -0.015098949742906037;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.6149935722351092) ) ) {
                      result[0] += -0.002272363267140622;
                    } else {
                      if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.00814361071642319;
                      } else {
                        result[0] += -0.08568296512731621;
                      }
                    }
                  } else {
                    result[0] += -0.045599469562458306;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += -0.002688659009289411;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.832297801971436435) ) ) {
                        result[0] += -0.07886528445973108;
                      } else {
                        result[0] += -0.015287565887375176;
                      }
                    } else {
                      result[0] += -0.000323265814097103;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += 0.005985338817849372;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                          result[0] += -0.024791522189411636;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                            result[0] += -0.03455496315830767;
                          } else {
                            result[0] += 0.005796920715395436;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.012042151668118338;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.494223117828370029) ) ) {
                    result[0] += 0.048104407070441224;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.457435131072998935) ) ) {
                      result[0] += 0.004473051221912173;
                    } else {
                      result[0] += -0.16068330483690998;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
            result[0] += -0.012054257660189074;
          } else {
            result[0] += -0.0029058135660437814;
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.002230406005819704;
        } else {
          result[0] += -0.04738040680367607;
        }
      }
    } else {
      result[0] += 0.00023861501458686307;
    }
  } else {
    result[0] += 0.09084217851926912;
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
        result[0] += 0.0035439328576569347;
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.680079460144043857) ) ) {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.009381002253218522;
            } else {
              result[0] += -0.04854439195814521;
            }
          } else {
            result[0] += -0.0019850683501019682;
          }
        } else {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.0003249929584226531;
          } else {
            result[0] += -0.033313679709990066;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
            result[0] += 0.03800875304188923;
          } else {
            result[0] += -0.08258995388161311;
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.014888973417977267;
                } else {
                  result[0] += 0.0012899199842635054;
                }
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0075359503569196575;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.0008872809544248897;
                        } else {
                          result[0] += -0.04983881585941415;
                        }
                      }
                    } else {
                      result[0] += -0.03453862180758293;
                    }
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                        result[0] += -0.0468240188009873;
                      } else {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                          result[0] += -0.02326119891041393;
                        } else {
                          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.346512794494629794) ) ) {
                            result[0] += -0.020996347362296094;
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.875080585479737216) ) ) {
                              result[0] += 0.015926467557975797;
                            } else {
                              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                                result[0] += 0.0704002099795358;
                              } else {
                                result[0] += 0.02427524159636263;
                              }
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += 0.009712649206525;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.158124685287476474) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
                        result[0] += 0.010863261407915114;
                      } else {
                        if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += -0.008872985339393994;
                        } else {
                          result[0] += -0.046625132879669545;
                        }
                      }
                    } else {
                      result[0] += 0.002886079075287231;
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.624251961708069292) ) ) {
                      result[0] += 0.0059487720495108535;
                    } else {
                      result[0] += -0.04205820922380693;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.0015776389475154523;
                } else {
                  result[0] += -0.04052540922670352;
                }
              } else {
                result[0] += 0.012732146317016444;
              }
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.859359741210938388) ) ) {
                  result[0] += 0.0012888506371261977;
                } else {
                  if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03619874700532021;
                  } else {
                    if ( LIKELY( !(data[34].missing != -1) || (data[34].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.03135260873481588;
                    } else {
                      result[0] += 0.025697612550403693;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.034252948722582;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.625819444656372958) ) ) {
                      result[0] += -0.0038030219269058483;
                    } else {
                      result[0] += 0.013374301216230622;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.126931190490723544) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.645421981811524326) ) ) {
                          result[0] += -0.03561624471527864;
                        } else {
                          result[0] += 0.0247438943662292;
                        }
                      } else {
                        result[0] += 0.022690051348363;
                      }
                    } else {
                      result[0] += 0.0011403776872553953;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += 0.0009704674547563899;
                      } else {
                        result[0] += -0.06363571293350763;
                      }
                    } else {
                      result[0] += 0.004937598533362071;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.901557445526123935) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.12901973724365412) ) ) {
                  result[0] += 0.0031884210151714057;
                } else {
                  result[0] += 0.0748926940904703;
                }
              } else {
                result[0] += 0.029966365892349883;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.016982157676156927;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                  result[0] += 0.02174230679469022;
                } else {
                  result[0] += -0.007892975005980896;
                }
              }
            } else {
              result[0] += 0.0003149721261715819;
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.276966691017151323) ) ) {
                  result[0] += -0.03901052385030104;
                } else {
                  result[0] += -0.08904963418203135;
                }
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.901921629905701128) ) ) {
                  result[0] += 0.021752165060202142;
                } else {
                  result[0] += -0.03320656418983635;
                }
              }
            } else {
              result[0] += -0.0021207088494465186;
            }
          }
        } else {
          result[0] += 0.008644900850399281;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += -0.0003847764895952604;
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
        result[0] += -0.0034826946551331154;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.07217657009707598;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
              result[0] += -0.015091412174074432;
            } else {
              result[0] += -0.07782924587558566;
            }
          }
        } else {
          result[0] += -0.016064293173451424;
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.942744255065918857) ) ) {
          result[0] += -0.005467543051167721;
        } else {
          result[0] += -0.02562727940284498;
        }
      } else {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
              result[0] += -0.023515606799320434;
            } else {
              result[0] += 0.018049601437740172;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
              result[0] += -0.0006782292904283628;
            } else {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.014722428035768174;
              } else {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.041101936358265756;
                } else {
                  result[0] += -0.0698125121362761;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.745876312255860263) ) ) {
            result[0] += 0.011202589788681844;
          } else {
            result[0] += -0.06661979357110924;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          result[0] += -0.05512502386503846;
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
              result[0] += 0.001868574846300795;
            } else {
              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                result[0] += -0.003368019285880902;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                  result[0] += -0.02531892487642397;
                } else {
                  result[0] += 0.013177183396327044;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.01046385008039719;
              } else {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.0883522033691424) ) ) {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.0024898911066097663;
                    } else {
                      result[0] += 0.011048519406259835;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                      result[0] += 0.01968532346893966;
                    } else {
                      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                        result[0] += -0.039993396678484554;
                      } else {
                        result[0] += -0.005862728374961579;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0032199053685869686;
                }
              }
            } else {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.014788627624512607) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)2.962127923965454546) ) ) {
                        result[0] += 0.031655136516168685;
                      } else {
                        result[0] += -0.00032455885628466326;
                      }
                    } else {
                      result[0] += -0.012538420477301086;
                    }
                  } else {
                    result[0] += -0.01695622741266158;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                    result[0] += -0.027761001330090542;
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.108966588973999912) ) ) {
                          result[0] += 0.041746986759112974;
                        } else {
                          result[0] += -0.009044985320155276;
                        }
                      } else {
                        result[0] += -0.02272685080942057;
                      }
                    } else {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.51517200469970881) ) ) {
                              result[0] += 0.0039122932984346345;
                            } else {
                              result[0] += 0.027602851539908192;
                            }
                          } else {
                            result[0] += -0.0018636059531085887;
                          }
                        } else {
                          result[0] += 0.02522249286495525;
                        }
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.33441734313965021) ) ) {
                          result[0] += 0.017247965821575867;
                        } else {
                          result[0] += 0.05935551476162029;
                        }
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.21245336532592951) ) ) {
                  if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.002332482082370732;
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.913499355316162998) ) ) {
                        result[0] += -0.06210029555196122;
                      } else {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                          result[0] += -0.03636068055439121;
                        } else {
                          result[0] += 0.02059798861628746;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.002655105456887262;
                      } else {
                        result[0] += -0.05431187490846778;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.038607705067684205;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                      result[0] += -0.02763262940099498;
                    } else {
                      result[0] += 0.018872313393687404;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            result[0] += 6.698983690455356e-05;
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.036215237346097606;
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                      result[0] += 0.10703751424943025;
                    } else {
                      result[0] += -0.007720498694061511;
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                          result[0] += -0.005662224775484058;
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.333273410797120029) ) ) {
                            result[0] += 0.08525056403616199;
                          } else {
                            result[0] += 0.017511176045970692;
                          }
                        }
                      } else {
                        result[0] += -0.024356491127285357;
                      }
                    } else {
                      result[0] += 0.02026019309825252;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                    result[0] += 0.012645926809756923;
                  } else {
                    result[0] += -0.009368234419520258;
                  }
                }
              } else {
                result[0] += -0.003360402030644758;
              }
            }
          }
        } else {
          result[0] += 0.007931756444363447;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += -0.0003705695190776438;
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
        result[0] += -0.006079579992375055;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.198464870452881303) ) ) {
            result[0] += -0.07146863264301744;
          } else {
            result[0] += 0.07646930386679952;
          }
        } else {
          result[0] += -0.020581424542950238;
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
    if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.450390577316285068) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.232417106628418857) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.556798219680787021) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.001563087449759225;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.870983839035034624) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.1281237259110376;
                      } else {
                        result[0] += -0.028620599120446122;
                      }
                    } else {
                      result[0] += 0.02175693775800427;
                    }
                  }
                } else {
                  result[0] += -0.020743105258831242;
                }
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                    result[0] += -0.024199544923750347;
                  } else {
                    result[0] += -0.002163363100203127;
                  }
                } else {
                  result[0] += 0.009591501903225004;
                }
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += -0.051954260360356046;
                } else {
                  result[0] += 0.00124436766556633;
                }
              } else {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.000639670252743921;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                    if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += -0.014083505650817752;
                    } else {
                      result[0] += 0.03437949516193007;
                    }
                  } else {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0300452048192016;
                    } else {
                      result[0] += 0.006733972613582096;
                    }
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.0003171014201215304;
            } else {
              if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.07089868517462145;
                  } else {
                    result[0] += -0.01934179756977321;
                  }
                } else {
                  result[0] += -0.017716312592794502;
                }
              } else {
                if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                    result[0] += -0.16465637916212564;
                  } else {
                    result[0] += -0.010140587970338634;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.249904870986938921) ) ) {
                      result[0] += 0.1585207200158755;
                    } else {
                      result[0] += -0.002182770057569384;
                    }
                  } else {
                    result[0] += 0.0697750524920147;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.747798204421997958) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
                    result[0] += 0.09144856453676636;
                  } else {
                    result[0] += 0.015677586206439623;
                  }
                } else {
                  result[0] += -0.02001326535180603;
                }
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2252.000000000000455) ) ) {
                  result[0] += 0.10891562835105123;
                } else {
                  result[0] += 0.0225244848116845;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += -0.12934858355592643;
                } else {
                  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.04192889250284466;
                  } else {
                    result[0] += 0.03100569747204265;
                  }
                }
              } else {
                result[0] += 0.041987742526915874;
              }
            }
          } else {
            result[0] += 0.13071778883984037;
          }
        }
      } else {
        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
            if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.015841438127251885;
              } else {
                result[0] += 0.12829093044893722;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.791781663894654208) ) ) {
                result[0] += -0.09630043008054782;
              } else {
                result[0] += 0.007844251814012415;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.438968896865845615) ) ) {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.009725655131999196;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.799612998962403232) ) ) {
                      result[0] += -0.04481198523396354;
                    } else {
                      result[0] += -0.15568788120315982;
                    }
                  } else {
                    result[0] += -0.010850017196927944;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.558514595031739169) ) ) {
                  result[0] += -0.17499948443071192;
                } else {
                  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.12586924964215726;
                  } else {
                    result[0] += -0.008604998009592316;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.11581762533477763;
              } else {
                result[0] += -0.008386233754033263;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            result[0] += -0.01347555851924569;
          } else {
            result[0] += 0.11792532142893027;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
          result[0] += -0.11303155189478889;
        } else {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
            result[0] += 0.005063812413161242;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
              result[0] += -0.13756473704335176;
            } else {
              result[0] += 0.03014741544865563;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.722943305969239169) ) ) {
            result[0] += 0.1392879741820699;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += 0.013598173123334814;
            } else {
              result[0] += 0.09883024793975655;
            }
          }
        } else {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            result[0] += 0.05812115332408662;
          } else {
            if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.015537291181750627;
            } else {
              result[0] += 0.020894242594314612;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.022538185119630683) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.412779808044434482) ) ) {
        result[0] += -0.0026637627516083204;
      } else {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.471622467041016513) ) ) {
          result[0] += -0.19989815512118514;
        } else {
          result[0] += 0.0022274483648332165;
        }
      }
    } else {
      result[0] += -0.13539477087800794;
    }
  }
  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
    result[0] += -0.0002556291092484808;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.827628135681153232) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.315225601196289951) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.929516315460205966) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                        result[0] += 0.00822371967232095;
                      } else {
                        result[0] += -0.0503222721134827;
                      }
                    } else {
                      if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.11066316778206729;
                        } else {
                          result[0] += -0.00435249201454528;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += 0.0010241662147253737;
                          } else {
                            result[0] += 0.031840953800407676;
                          }
                        } else {
                          result[0] += 0.009835770532737856;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.00038635145400278995;
                  }
                } else {
                  if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                    result[0] += -0.04955714971448164;
                  } else {
                    result[0] += 0.023457219014243136;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                    result[0] += -0.0076299402411638025;
                  } else {
                    result[0] += 0.010530160835092492;
                  }
                } else {
                  result[0] += -0.0081078441941745;
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.03320790452446307;
                    } else {
                      result[0] += 0.00021117396876427997;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                      result[0] += -0.012517111314539099;
                    } else {
                      result[0] += 0.004604388388096193;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.08329235019506916;
                  } else {
                    result[0] += -0.015632197078944365;
                  }
                }
              } else {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                  result[0] += -0.010077042013496108;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.02727941574955043;
                  } else {
                    result[0] += -0.03135167143966314;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += -0.03766426523020226;
            } else {
              result[0] += -0.007106599621705337;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                  result[0] += 0.0005785776350939379;
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.901921629905701128) ) ) {
                    result[0] += -0.0311961131046192;
                  } else {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += 0.013675929207824523;
                    } else {
                      if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                        result[0] += -0.036468700255340145;
                      } else {
                        result[0] += 0.006021838942882885;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                  result[0] += -0.029199444909215833;
                } else {
                  result[0] += -0.07501324787751519;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.490982532501221591) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.046284758933677096;
                } else {
                  result[0] += -0.07876506976499222;
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += -0.00308112867053725;
                } else {
                  result[0] += 0.01781686630752412;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.00880657259382075;
              } else {
                if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.047201039015084774;
                } else {
                  result[0] += 0.008956476881036917;
                }
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.002474950630394924;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                  if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.03420138359069913) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)208.0000000000000284) ) ) {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.979851245880127841) ) ) {
                            result[0] += 0.0032469518522104304;
                          } else {
                            result[0] += 0.05394865858847972;
                          }
                        } else {
                          if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += -0.07123996589554821;
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
                              result[0] += 0.004580355749162208;
                            } else {
                              result[0] += -0.04753650679047964;
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.338562726974488193) ) ) {
                          result[0] += 0.00748454326914659;
                        } else {
                          result[0] += 0.040109146710436326;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.03794336773042914;
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.730421066284180576) ) ) {
                          result[0] += 0.07056211509272448;
                        } else {
                          result[0] += -0.06894804336115846;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                      result[0] += -0.006816137757616799;
                    } else {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        result[0] += -0.010452878152335967;
                      } else {
                        result[0] += -0.06304533474008105;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.18732333183288663) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.569529533386231357) ) ) {
                      result[0] += 0.011281012807005858;
                    } else {
                      result[0] += -0.0340780762296482;
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                        result[0] += -0.010353607963508567;
                      } else {
                        result[0] += 0.07533754730968328;
                      }
                    } else {
                      result[0] += 0.007481874779901927;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.014788627624512607) ) ) {
            result[0] += 0.008446818002670163;
          } else {
            result[0] += 0.041144126608442126;
          }
        } else {
          result[0] += -0.009836575620109495;
        }
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
        result[0] += -0.014424100407293895;
      } else {
        result[0] += 0.01398228058721917;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.968589782714844638) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.00013526035616508075;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
            result[0] += 0.0014759333345989445;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.274755001068116123) ) ) {
                  result[0] += 0.04076467613070243;
                } else {
                  result[0] += -0.000458480883299131;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.134879350662232333) ) ) {
                  result[0] += -0.0056151741185543405;
                } else {
                  result[0] += 0.008093490337598151;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.6134371757507342) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.123651981353760654) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                          result[0] += 0.0009901655819805138;
                        } else {
                          result[0] += -0.03712931944230228;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.689592361450196201) ) ) {
                          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                            result[0] += -0.0027868379118254276;
                          } else {
                            if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.0353790224889882;
                            } else {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                                result[0] += 0.026813865955856678;
                              } else {
                                result[0] += -0.03379707577091775;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                            result[0] += -0.037995918271207685;
                          } else {
                            result[0] += -0.00725029688418591;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.039663550605897685;
                        } else {
                          result[0] += -0.00044299413660590546;
                        }
                      } else {
                        result[0] += 0.00442032138104718;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.700190305709839755) ) ) {
                      result[0] += 0.012679363330741301;
                    } else {
                      result[0] += -0.06239561771235343;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.04363514966593783;
                  } else {
                    result[0] += 0.0001055605830587027;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.036670446395874912) ) ) {
                          result[0] += 0.001477686982357926;
                        } else {
                          result[0] += 0.009851723052071798;
                        }
                      } else {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
                            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.02377576331246434;
                            } else {
                              result[0] += 0.13640874212933407;
                            }
                          } else {
                            result[0] += -0.036436718454402404;
                          }
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                                result[0] += -0.13130740856180673;
                              } else {
                                result[0] += -0.0014691864261913003;
                              }
                            } else {
                              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.497866153717041238) ) ) {
                                result[0] += 0.13901503724213846;
                              } else {
                                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                                  result[0] += -0.04762442346751293;
                                } else {
                                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
                                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                                      result[0] += -0.0025782291512322945;
                                    } else {
                                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.305786132812500888) ) ) {
                                        result[0] += 0.04747330236298603;
                                      } else {
                                        result[0] += 0.01247913293610816;
                                      }
                                    }
                                  } else {
                                    result[0] += -0.03723774907565638;
                                  }
                                }
                              }
                            }
                          } else {
                            result[0] += -0.008655726811658047;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.10057943603964353;
                      } else {
                        result[0] += 0.020475382115931376;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.06511284779295906;
                    } else {
                      result[0] += 0.01609043327557996;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.012003979824742661;
                      } else {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                          result[0] += 0.000945589482219789;
                        } else {
                          result[0] += 0.05460727081948317;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.579839229583741123) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                          result[0] += -0.002026913004848727;
                        } else {
                          result[0] += -0.03083302825915582;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
                          result[0] += 0.07074029578389096;
                        } else {
                          result[0] += -0.00870188407429147;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.0023624478175766385;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
            result[0] += -0.003633154412253625;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.008761351046593839;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                result[0] += -0.039023398577264556;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.505334615707398349) ) ) {
                  result[0] += -0.029036799714739443;
                } else {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.017714808081565805;
                  } else {
                    result[0] += 0.08839625932010475;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.015298734394748778;
    }
  } else {
    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
          result[0] += 0.008446452666904237;
        } else {
          result[0] += -0.08779447640816418;
        }
      } else {
        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.174569487571716753) ) ) {
            result[0] += -0.0025808623390540146;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.259080410003662998) ) ) {
                result[0] += -0.0011721265372156078;
              } else {
                result[0] += 0.06181319473336247;
              }
            } else {
              result[0] += 0.036256053393804825;
            }
          }
        } else {
          result[0] += -0.08388093290170316;
        }
      }
    } else {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
        result[0] += 0.024263048821225167;
      } else {
        result[0] += -0.008311556883204011;
      }
    }
  }
  if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
    result[0] += 0.00018629045858811175;
  } else {
    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.47727227210998624) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.0027421690805922642;
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      result[0] += -0.05354482211976457;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
                          result[0] += -0.04383752967232943;
                        } else {
                          result[0] += 0.005308984600823099;
                        }
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.552491664886475498) ) ) {
                          result[0] += -0.014450110607204407;
                        } else {
                          result[0] += -0.0572859354119933;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.2692751884460467) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                          result[0] += 0.0391435173465063;
                        } else {
                          result[0] += -0.034289301955267054;
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                          result[0] += 0.08114079587726927;
                        } else {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.463808774948121005) ) ) {
                            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                              result[0] += -0.028829903112523727;
                            } else {
                              result[0] += 0.05535287082506936;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                              result[0] += -0.08020224152405113;
                            } else {
                              result[0] += 0.026924351042983703;
                            }
                          }
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                          result[0] += -0.05518297736052036;
                        } else {
                          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                              result[0] += 0.018292259270545196;
                            } else {
                              result[0] += 0.051934163973530234;
                            }
                          } else {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.13839721679687678) ) ) {
                              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.668153762817383701) ) ) {
                                result[0] += -0.06153152753542292;
                              } else {
                                result[0] += 0.007593293259435737;
                              }
                            } else {
                              result[0] += 0.018743274936425865;
                            }
                          }
                        }
                      } else {
                        result[0] += -0.046555366072204785;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                    result[0] += -0.03450915763972886;
                  } else {
                    result[0] += -0.01184879591227894;
                  }
                }
              }
            } else {
              result[0] += 0.01349073609428733;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.004790294696463734;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.469231128692627841) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.182021141052246982) ) ) {
                          result[0] += -0.01302011010159456;
                        } else {
                          result[0] += 0.0015612863134681263;
                        }
                      } else {
                        result[0] += 0.03225628560553986;
                      }
                    } else {
                      result[0] += 0.10437573534769273;
                    }
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.037340745736335355;
                        } else {
                          result[0] += 0.0025305467579466545;
                        }
                      } else {
                        result[0] += -0.04016678837215479;
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += 0.0009718710488916659;
                      } else {
                        result[0] += 0.0649809534253146;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.02998451639065281;
                  } else {
                    result[0] += -0.006551360525535126;
                  }
                }
              }
            } else {
              result[0] += 0.00038159185382815794;
            }
          }
        } else {
          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += -0.10088773618532812;
              } else {
                result[0] += -0.013661418673876276;
              }
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.628996372222901279) ) ) {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                    result[0] += -0.01977867563018997;
                  } else {
                    result[0] += -0.0627422077229939;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
                    result[0] += -0.012162257280154729;
                  } else {
                    result[0] += 0.058246330566051774;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                  result[0] += -0.028664518649074402;
                } else {
                  result[0] += 0.011766683719556865;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                  result[0] += 0.0014285032747766266;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                    result[0] += 0.01475429189721763;
                  } else {
                    result[0] += 0.07776816372404272;
                  }
                }
              } else {
                result[0] += -0.018420918206272972;
              }
            } else {
              result[0] += -0.016131618435538755;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04359999574764291;
            } else {
              result[0] += -0.00012002546817600792;
            }
          } else {
            result[0] += -0.025449846632201706;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += -0.0008165834405672094;
            } else {
              result[0] += -0.024171674223524696;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.921924352645874468) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.208590507507325107) ) ) {
                if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                  result[0] += 0.008293587898128596;
                } else {
                  result[0] += -0.0025116684850141225;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.237347126007080966) ) ) {
                    result[0] += 0.008380552798840332;
                  } else {
                    result[0] += -0.06751671660663963;
                  }
                } else {
                  result[0] += 0.027184612039243234;
                }
              }
            } else {
              result[0] += -0.02613272788226383;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += 0.005872253084959169;
      } else {
        result[0] += -0.03150040818734958;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.968589782714844638) ) ) {
      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
        result[0] += 0.00016222711558715785;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
              if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                  result[0] += 0.0004339237076046224;
                } else {
                  result[0] += 0.021989779130807355;
                }
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += 0.0391211134917294;
                  } else {
                    result[0] += -0.0036289185003744204;
                  }
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                      result[0] += -0.0034963548924069714;
                    } else {
                      result[0] += -0.18708114323456687;
                    }
                  } else {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.067782521247864214) ) ) {
                        result[0] += -0.027997973364911513;
                      } else {
                        result[0] += 0.010053247032732741;
                      }
                    } else {
                      result[0] += 0.1786516690342517;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                result[0] += -0.004361042516430523;
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += -0.01479229787456772;
                } else {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += -0.03637830880586978;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
                        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.53498554229736506) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            result[0] += 0.040238662109034554;
                          } else {
                            result[0] += 0.0006623186925244866;
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                            result[0] += 0.020496516554462607;
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                              result[0] += 0.0472394153708722;
                            } else {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += -0.005693387545399831;
                              } else {
                                result[0] += -0.10702292460737935;
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += 0.0035268408120920206;
                            } else {
                              result[0] += 0.0352978937313822;
                            }
                          } else {
                            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                              result[0] += -0.06453884953893756;
                            } else {
                              result[0] += -0.010958783874784002;
                            }
                          }
                        } else {
                          result[0] += -0.018491078210874968;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05539035797119318) ) ) {
                        result[0] += 0.004459255891005486;
                      } else {
                        result[0] += 0.020197140388361942;
                      }
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.087577104568482333) ) ) {
                          result[0] += -0.004936201644409199;
                        } else {
                          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                            result[0] += 0.01827055023649748;
                          } else {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.05539035797119318) ) ) {
                              result[0] += 0.0034585977109638354;
                            } else {
                              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                                result[0] += -0.00431471259874511;
                              } else {
                                result[0] += 0.01077744478487642;
                              }
                            }
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.602003335952759233) ) ) {
                          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                            result[0] += 0.005727182358705817;
                          } else {
                            result[0] += -0.008088614138156014;
                          }
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.016415807727674165;
                          } else {
                            result[0] += -0.017978742024119487;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
              result[0] += -0.06926760818116182;
            } else {
              result[0] += -0.010648853430018087;
            }
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.007014962033292563;
              } else {
                result[0] += -0.04985434600178897;
              }
            } else {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  result[0] += -0.06706560959232484;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11581277847290217) ) ) {
                        result[0] += 0.03560220073477378;
                      } else {
                        result[0] += -0.048030907685930144;
                      }
                    } else {
                      result[0] += 0.039085167329091536;
                    }
                  } else {
                    if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                      result[0] += 0.0033533271433414495;
                    } else {
                      result[0] += -0.015689504052537617;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.551017761230469638) ) ) {
                  result[0] += 0.017058641291962343;
                } else {
                  result[0] += -0.02651024110560604;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
              result[0] += -0.033225798830510335;
            } else {
              result[0] += -0.007845887949625542;
            }
          }
        }
      }
    } else {
      result[0] += -0.01501402425604505;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
          result[0] += 0.007502505714122743;
        } else {
          result[0] += -0.08089313982800328;
        }
      } else {
        result[0] += 0.010325861086731392;
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.339535951614380771) ) ) {
              result[0] += 0.004324436237081757;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                result[0] += 0.15872204801493384;
              } else {
                result[0] += 0.00027529721564922187;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += 0.1088200165013069;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                result[0] += 0.05705809103116249;
              } else {
                result[0] += 0.014243124460268528;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.412779808044434482) ) ) {
            result[0] += -0.0024070382341833322;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.46012759208679288) ) ) {
              result[0] += -0.19985156581819902;
            } else {
              result[0] += 0.0036889230272294233;
            }
          }
        }
      } else {
        result[0] += -0.07899534492448004;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.968589782714844638) ) ) {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
          result[0] += 0.00011485945468325815;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)14.26534414291382014) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
                      result[0] += -0.08356576525259636;
                    } else {
                      result[0] += 0.06701224082933789;
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.00335694773026937;
                    } else {
                      result[0] += -0.027148808218511622;
                    }
                  }
                } else {
                  result[0] += 0.058180269731555614;
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.09760948635644481;
                  } else {
                    result[0] += -0.00027022282836556953;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81278371810913264) ) ) {
                    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                        result[0] += 0.018326709058265483;
                      } else {
                        result[0] += -0.002602288815957264;
                      }
                    } else {
                      result[0] += -0.01976842187107458;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.48872375488281428) ) ) {
                        result[0] += -0.016464939441686796;
                      } else {
                        result[0] += 0.0315515911510647;
                      }
                    } else {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += -0.09738939176014007;
                      } else {
                        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.006501391484908044;
                        } else {
                          result[0] += 0.04304812206465322;
                        }
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.06614680162062672;
              } else {
                result[0] += -0.0080341852061732;
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11407232284546076) ) ) {
                result[0] += -0.01097257148236504;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.966329097747803623) ) ) {
                  result[0] += -0.05687872211675704;
                } else {
                  result[0] += -0.015246842861980518;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.003666865989713884;
                  } else {
                    result[0] += -0.07426447482943088;
                  }
                } else {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.0387635940833179;
                  } else {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                      result[0] += -0.11169632633815996;
                    } else {
                      result[0] += 0.0002452618588492706;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01812194811269513;
                  } else {
                    result[0] += 0.002721273684290938;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.2692751884460467) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                      result[0] += -0.01468511164224371;
                    } else {
                      result[0] += -0.043874107756898095;
                    }
                  } else {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.016193069291800467;
                    } else {
                      result[0] += -0.011902934800221823;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
            result[0] += 0.02706683430152006;
          } else {
            result[0] += -0.04983245050881158;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
            result[0] += -0.03701600219056921;
          } else {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.350257158279419833) ) ) {
                  result[0] += 0.031166818325747487;
                } else {
                  result[0] += -0.0832854430346491;
                }
              } else {
                result[0] += 0.052933614309268934;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.44140100479126021) ) ) {
                result[0] += -0.07836473728805192;
              } else {
                result[0] += 0.009624203559661813;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.01481427249068684;
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)7.998151302337647373) ) ) {
        result[0] += -0.018978152390458074;
      } else {
        result[0] += -0.08532119853941149;
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
        if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.339535951614380771) ) ) {
              if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.569433569908142534) ) ) {
                  result[0] += -0.08070863993317473;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                    result[0] += -0.05591848400279797;
                  } else {
                    result[0] += 0.008246471963378426;
                  }
                }
              } else {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += -0.020495427633322554;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.745876312255860263) ) ) {
                        result[0] += 0.10773023380685345;
                      } else {
                        result[0] += -0.02990277530876786;
                      }
                    } else {
                      result[0] += 0.0039125027397318105;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += -0.02697708642351039;
                  } else {
                    result[0] += 0.024089547533917188;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
                result[0] += 0.15769497144047237;
              } else {
                result[0] += 0.002022195175555231;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += 0.10971895342353158;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.31402075290679976) ) ) {
                  result[0] += 0.004378575295659354;
                } else {
                  result[0] += -0.05125908954383438;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                  result[0] += 0.0763592566077878;
                } else {
                  result[0] += 0.020857602448547483;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.412779808044434482) ) ) {
            result[0] += -0.0018391105453498197;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.46012759208679288) ) ) {
              result[0] += -0.19981177493535918;
            } else {
              result[0] += 0.004189847481210549;
            }
          }
        }
      } else {
        result[0] += -0.07238637741177432;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.968589782714844638) ) ) {
      if ( LIKELY( !(data[40].missing != -1) || (data[40].fvalue <= (double)1.500000000000000222) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)5.877672910690308505) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += 7.678185724660277e-05;
          } else {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              result[0] += -0.0003502332228551672;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                  result[0] += -0.009170750203462096;
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.207582950592041904) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += -0.013982941881203243;
                    } else {
                      result[0] += -0.05243842120309007;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.15100884437561124) ) ) {
                      result[0] += -0.06405795531182545;
                    } else {
                      result[0] += -0.22867893495568953;
                    }
                  }
                }
              } else {
                result[0] += 0.002973100076986985;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                result[0] += 0.10759113619881269;
              } else {
                result[0] += -0.03768697552483727;
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                result[0] += -0.09213010427662749;
              } else {
                result[0] += 0.020079478260575632;
              }
            }
          } else {
            result[0] += 0.10444871574908716;
          }
        }
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.349460363388062412) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.232009172439576083) ) ) {
              result[0] += 0.06987430113614967;
            } else {
              result[0] += -0.09996504549530423;
            }
          } else {
            result[0] += -0.043619850250100824;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.241249561309815341) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
              result[0] += -0.031241434151276296;
            } else {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)5.500000000000000888) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06112537226748559;
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.297262430191040927) ) ) {
                      result[0] += 0.02330428054535634;
                    } else {
                      result[0] += -0.10426381555029858;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                    result[0] += -0.05909998175659817;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.04325845156378628;
                    } else {
                      result[0] += -0.08989645626079479;
                    }
                  }
                }
              } else {
                result[0] += -0.040319056210566355;
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
              result[0] += -0.04917192542906233;
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.827628135681153232) ) ) {
                result[0] += 0.06686959259529013;
              } else {
                result[0] += 0.15561133877702327;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.14096093177795499) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.310776710510254794) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)10.50000000000000178) ) ) {
            result[0] += -0.01698215200426312;
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                  result[0] += -0.010881726621249673;
                } else {
                  result[0] += -0.1589704330673016;
                }
              } else {
                result[0] += 0.03036287226813186;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.447260618209839755) ) ) {
                  result[0] += -0.06846285252361398;
                } else {
                  result[0] += 0.02994253020673995;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += 0.1041077294372269;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                    result[0] += -0.015634024724625425;
                  } else {
                    result[0] += 0.10578333463016672;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)4.808102607727051669) ) ) {
            result[0] += -0.013659276094306075;
          } else {
            result[0] += -0.15614569747171836;
          }
        }
      } else {
        result[0] += -0.10526259793677417;
      }
    }
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)7.998151302337647373) ) ) {
        result[0] += -0.018979322269809102;
      } else {
        result[0] += -0.08513227064071123;
      }
    } else {
      if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)24.00000000000000355) ) ) {
        if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.03420138359069913) ) ) {
          if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.86282253265381037) ) ) {
              result[0] += -0.004268751927909208;
            } else {
              result[0] += -0.18484991313548355;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.9055976867675799) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += -0.004391742736429695;
              } else {
                result[0] += -0.11913631977459219;
              }
            } else {
              result[0] += 0.09309868704553302;
            }
          }
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
              if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += 0.05940480677331861;
              } else {
                result[0] += -0.020913356119484347;
              }
            } else {
              result[0] += 0.10995951808314014;
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.449861526489258257) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)12.00000000000000178) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                    result[0] += -0.0015088645786880029;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                        result[0] += -0.02098158550044867;
                      } else {
                        result[0] += -0.19937222934499305;
                      }
                    } else {
                      result[0] += -0.016719124518377176;
                    }
                  }
                } else {
                  result[0] += 0.019633232307996858;
                }
              } else {
                result[0] += -0.09160143790329854;
              }
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)2.500000000000000444) ) ) {
                    result[0] += 0.013739187358129719;
                  } else {
                    result[0] += 0.08241898169031309;
                  }
                } else {
                  result[0] += 0.16328721553860576;
                }
              } else {
                result[0] += 0.015702524331208376;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.412779808044434482) ) ) {
          if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.07145882250736038;
          } else {
            result[0] += -0.0037031863004339627;
          }
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.46012759208679288) ) ) {
            result[0] += -0.19976075959971565;
          } else {
            result[0] += 0.004665541582657533;
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.037492996647503954;
      } else {
        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              result[0] += -0.025848971709021507;
            } else {
              result[0] += 0.0007167544155074619;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.01395750045776456) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.583132982254028764) ) ) {
                  result[0] += -0.02475599619402403;
                } else {
                  result[0] += -0.17894077151563154;
                }
              } else {
                result[0] += -0.06412573743474947;
              }
            } else {
              result[0] += -0.0033712645461114674;
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
            if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.998158693313599077) ) ) {
                if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                  result[0] += -0.007070982583419209;
                } else {
                  result[0] += -0.14531636630645825;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += 0.14899655269147086;
                } else {
                  result[0] += 0.016995212405389704;
                }
              }
            } else {
              result[0] += 0.01417808523304329;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.02173149086261205;
            } else {
              result[0] += -0.0036843972378980817;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
          if ( UNLIKELY(  (data[38].missing != -1) && (data[38].fvalue <= (double)-1.00000001800250948e-35) ) ) {
            result[0] += 0.0017362143604497055;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04098124866160072;
            } else {
              result[0] += -0.009739480476247871;
            }
          }
        } else {
          if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.008002718572263789;
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.592359304428101474) ) ) {
                    result[0] += 0.002222451249336057;
                  } else {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                      result[0] += 0.060863428731628415;
                    } else {
                      result[0] += 0.022067886932008583;
                    }
                  }
                } else {
                  result[0] += 0.0047003156294744535;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.166635274887085849) ) ) {
                result[0] += -0.11602540675367688;
              } else {
                result[0] += 0.1228189467506362;
              }
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.235757827758790839) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.549068689346314365) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.869292974472046787) ) ) {
                      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                          result[0] += -0.02341219907095041;
                        } else {
                          result[0] += 0.0027053351807568293;
                        }
                      } else {
                        result[0] += 0.011386788422978007;
                      }
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
                        result[0] += 0.014835247909592534;
                      } else {
                        result[0] += -0.012483245821557826;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.909254074096680576) ) ) {
                        result[0] += -0.009523124895684383;
                      } else {
                        result[0] += 0.0006100432224970438;
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.068990230560303623) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
                          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            result[0] += 0.001047973988277623;
                          } else {
                            result[0] += -0.010082009274283293;
                          }
                        } else {
                          result[0] += 0.00628086694410504;
                        }
                      } else {
                        result[0] += -0.018367790588906596;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.725620865821838823) ) ) {
                        result[0] += -0.0015918614000240841;
                      } else {
                        if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                          result[0] += -0.036297997858237015;
                        } else {
                          result[0] += -0.01177862101097177;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
                        result[0] += -0.04126945454029556;
                      } else {
                        result[0] += 0.0049396685620447835;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.068990230560303623) ) ) {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.985194206237793857) ) ) {
                        result[0] += -0.01242626492805596;
                      } else {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.901921629905701128) ) ) {
                          result[0] += 0.012854495957394622;
                        } else {
                          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
                            result[0] += 0.01588288251356038;
                          } else {
                            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.09641252755174981;
                            } else {
                              result[0] += -0.007513935184307436;
                            }
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                        result[0] += 0.031525700664088684;
                      } else {
                        result[0] += 0.00799590116583975;
                      }
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0344577035722723;
                  } else {
                    result[0] += -0.010081745032422124;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.003838300704956943) ) ) {
                      result[0] += -0.0026879513098125867;
                    } else {
                      result[0] += 0.016910944596131555;
                    }
                  } else {
                    if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.11766508942083248;
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                        result[0] += -0.015515591917052372;
                      } else {
                        result[0] += 0.05083507257580625;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.0030952935546334406;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.8154783248901385) ) ) {
          if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
              result[0] += -0.05105253290116786;
            } else {
              result[0] += -0.002360327807654539;
            }
          } else {
            result[0] += 0.01728310889851711;
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)9.500000000000001776) ) ) {
            result[0] += 0.01780179052000371;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.881510615348816362) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.93885374069213956) ) ) {
                result[0] += -0.1020785267791729;
              } else {
                result[0] += 0.09612364438182852;
              }
            } else {
              result[0] += 0.17337541613222918;
            }
          }
        }
      }
    }
  } else {
    result[0] += 0.00021492944707530634;
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.982575893402101386) ) ) {
          result[0] += 0.001560813981314436;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.03395731466386628;
                } else {
                  result[0] += -0.030315436350930636;
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.0025736655310809133;
                    } else {
                      result[0] += -0.056061933846378725;
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += -0.008595622366158108;
                        } else {
                          result[0] += 0.0400885594298096;
                        }
                      } else {
                        result[0] += -0.10553156848191336;
                      }
                    } else {
                      result[0] += -0.0442186784445429;
                    }
                  }
                } else {
                  result[0] += 0.0016648728975674682;
                }
              }
            } else {
              result[0] += 0.0025994953710162466;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
              result[0] += -0.0011566788038235081;
            } else {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                result[0] += -0.005330062126693296;
              } else {
                if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                  result[0] += -0.014777562727443863;
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0645103649059287;
                  } else {
                    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.03726578474366823;
                    } else {
                      result[0] += -0.045859021760726;
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
              result[0] += -0.001302013794053332;
            } else {
              result[0] += -0.019649210820464665;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.31402075290679976) ) ) {
              result[0] += -0.031063885391728704;
            } else {
              result[0] += 0.013978221618538523;
            }
          }
        } else {
          if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.70956039428711115) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13022470474243342) ) ) {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                          result[0] += 0.014514921897514405;
                        } else {
                          result[0] += -0.014666195934438199;
                        }
                      } else {
                        result[0] += -0.03499925723964993;
                      }
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.08921853975771063;
                      } else {
                        result[0] += -0.013667338848023776;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                      result[0] += -0.005291871659740878;
                    } else {
                      result[0] += 0.017640530699817965;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.6134371757507342) ) ) {
                    result[0] += -0.05189349208039887;
                  } else {
                    result[0] += -0.015214832536308379;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                      result[0] += 0.0069006295521765225;
                    } else {
                      result[0] += 0.02990986018338314;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
                      result[0] += -0.02915817556757523;
                    } else {
                      result[0] += 0.012843399818904935;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.388237953186036044) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.757321834564209873) ) ) {
                          result[0] += -0.05614661333993124;
                        } else {
                          result[0] += 0.004114032980684179;
                        }
                      } else {
                        result[0] += 0.02739885488114064;
                      }
                    } else {
                      result[0] += 0.04420743766195138;
                    }
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.007174260331412517;
                      } else {
                        result[0] += -0.15576793383256024;
                      }
                    } else {
                      result[0] += 0.009163940406046808;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.012425303902322287;
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    result[0] += -0.029960151868056836;
                  } else {
                    result[0] += -0.08188606465895756;
                  }
                } else {
                  result[0] += -0.010144315928480606;
                }
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.09228263781937614;
                } else {
                  result[0] += -0.0053747236219364065;
                }
              }
            } else {
              if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)4.290553331375122958) ) ) {
                  result[0] += 0.008555803830415697;
                } else {
                  result[0] += 0.10767954953445753;
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.0016132434495286557;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                      result[0] += -0.06328719330442728;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.318498134613038886) ) ) {
                        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                          result[0] += 0.16974766293682186;
                        } else {
                          result[0] += 0.0180375853228876;
                        }
                      } else {
                        result[0] += 0.1042621614943188;
                      }
                    }
                  }
                } else {
                  result[0] += -0.008046368646362422;
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        result[0] += -0.00043950477449295383;
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.917405366897583452) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                  result[0] += -0.00941645250590405;
                } else {
                  result[0] += -0.055127597043709925;
                }
              } else {
                result[0] += 0.002776014930681465;
              }
            } else {
              result[0] += 0.0026023345710500575;
            }
          } else {
            result[0] += -0.021562035230216388;
          }
        } else {
          result[0] += -0.010831185907488933;
        }
      }
    }
  } else {
    result[0] += 0.11263710602407069;
  }
  if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
    result[0] += -0.0005614321401117131;
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.011523246765138495) ) ) {
        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
          result[0] += -0.0033281678156963108;
        } else {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.006463071939893071;
            } else {
              result[0] += -0.002719397981097462;
            }
          } else {
            if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
                result[0] += -0.002136161292448334;
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += 0.013360563893745809;
                  } else {
                    result[0] += -0.02475835022906686;
                  }
                } else {
                  result[0] += -0.020524324359023557;
                }
              }
            } else {
              result[0] += 0.007762553492896078;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
            if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                  result[0] += -0.007361184095365321;
                } else {
                  result[0] += -0.06929833249404634;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += 0.0024598299271452392;
                    } else {
                      if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += 0.041176530534544724;
                      } else {
                        result[0] += -0.036440419680845854;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.547126770019532138) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                        result[0] += 0.02698464869733549;
                      } else {
                        result[0] += 0.1758762482697509;
                      }
                    } else {
                      if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                        result[0] += 0.15826356719517332;
                      } else {
                        result[0] += 0.020245911079697778;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.86282253265381037) ) ) {
                      result[0] += 0.01873422404492849;
                    } else {
                      result[0] += -0.03819557783826725;
                    }
                  } else {
                    result[0] += -0.010441004702509568;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += -0.00835538668064381;
              } else {
                result[0] += 0.009118967637414701;
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                  result[0] += -0.004247077768738485;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.038511063858872846;
                  } else {
                    result[0] += -0.011111213999117492;
                  }
                }
              } else {
                result[0] += 0.003927005702560859;
              }
            } else {
              result[0] += -0.02707546936382881;
            }
          }
        } else {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += 0.01684878658176751;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                    result[0] += -1.6090077203435234e-05;
                  } else {
                    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.03621909408664303;
                    } else {
                      result[0] += -0.010595457941640159;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.006381289325295681;
                } else {
                  if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                          result[0] += -0.003934990915501568;
                        } else {
                          result[0] += 0.04121127289297592;
                        }
                      } else {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.453179836273194248) ) ) {
                          result[0] += 0.02330583298057151;
                        } else {
                          result[0] += -0.005749313939125519;
                        }
                      }
                    } else {
                      result[0] += -0.012579987773774151;
                    }
                  } else {
                    result[0] += 0.0010853367480659552;
                  }
                }
              }
            } else {
              result[0] += -0.042696093286524704;
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += 0.006946129916755736;
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
                    result[0] += -0.017501619361019925;
                  } else {
                    result[0] += 0.0010743139233843695;
                  }
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                  result[0] += 0.003650369457520117;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.170116901397705966) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.78560066223144709) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                        result[0] += 0.043571994855710644;
                      } else {
                        result[0] += -0.006349484948600083;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                        result[0] += -0.016502352850564145;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.043341875076294833) ) ) {
                            result[0] += 0.027942566507927032;
                          } else {
                            result[0] += 0.09708444299818536;
                          }
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.015060318504304438;
                          } else {
                            result[0] += 0.09684052087263745;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += 0.06125413948766063;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.20949268341064631) ) ) {
                  result[0] += 0.003142190282317836;
                } else {
                  result[0] += 0.04779201934264354;
                }
              } else {
                result[0] += 0.05399257390251597;
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.31402075290679976) ) ) {
        if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
          result[0] += 0.0018985811163084083;
        } else {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.05594554699141279;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
              result[0] += 0.087057746348902;
            } else {
              result[0] += -0.011978867837091877;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            result[0] += -0.04138041730658564;
          } else {
            result[0] += 0.029906959084217234;
          }
        } else {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
            result[0] += -0.02431382901675298;
          } else {
            result[0] += 0.01120961717999752;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.540854334831238237) ) ) {
          result[0] += 0.000961267352481765;
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
              result[0] += 0.000756954143783289;
            } else {
              if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += 0.004355271747914619;
                  } else {
                    result[0] += -0.03661038675293411;
                  }
                } else {
                  result[0] += 0.0009925143706070716;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.122815132141115058) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.90474271774292081) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                        result[0] += 0.002748365042654878;
                      } else {
                        result[0] += 0.08488241993957313;
                      }
                    } else {
                      result[0] += -0.07661349163201735;
                    }
                  } else {
                    result[0] += -0.028785370350774143;
                  }
                } else {
                  result[0] += -0.060204918870877436;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                result[0] += 0.008612045857976118;
              } else {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.10232371409096902;
                } else {
                  result[0] += 0.01846039991887187;
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                result[0] += -0.06937302556024846;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)6.031101226806641513) ) ) {
                  result[0] += -0.0011195929170925978;
                } else {
                  result[0] += -0.19270554094542192;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.198464870452881303) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.045119860749965844;
                } else {
                  result[0] += -0.013555295103580249;
                }
              } else {
                result[0] += -0.007670753467381339;
              }
            } else {
              if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.551071166992188388) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                          result[0] += 0.0032921079686460404;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.95056009292602717) ) ) {
                            result[0] += -0.02913450414786857;
                          } else {
                            result[0] += 0.0006297513717232777;
                          }
                        }
                      } else {
                        result[0] += 0.0020225604783708612;
                      }
                    } else {
                      result[0] += 0.01148271173044331;
                    }
                  } else {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.831997871398927558) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
                        result[0] += -0.006047318831281918;
                      } else {
                        result[0] += 0.07172165080225025;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.58491539955139249) ) ) {
                        result[0] += 0.01765312259290204;
                      } else {
                        result[0] += 0.09842989429292927;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                      result[0] += -0.034632973723923745;
                    } else {
                      result[0] += -0.0027687993249817977;
                    }
                  } else {
                    result[0] += -0.06414425652823995;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                  if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                    result[0] += -0.004197745459733131;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
                      result[0] += 0.06894326728960543;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
                        result[0] += -0.060143961866856124;
                      } else {
                        result[0] += -0.018919166876220004;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)9.329766750335695136) ) ) {
                    result[0] += -0.012445166737141118;
                  } else {
                    result[0] += 0.006406182482828303;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.909254074096680576) ) ) {
              if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.01102107653191384;
                } else {
                  result[0] += 0.01126708330272029;
                }
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += -0.03166290955313239;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                      result[0] += 0.007594504076664891;
                    } else {
                      result[0] += -0.016972170725007098;
                    }
                  } else {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.938058137893677646) ) ) {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.88192772865295499) ) ) {
                          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                              result[0] += -0.03778893489998009;
                            } else {
                              result[0] += 0.011125144094562662;
                            }
                          } else {
                            result[0] += 0.004819368399926463;
                          }
                        } else {
                          result[0] += 0.00730514581111188;
                        }
                      } else {
                        result[0] += 0.016046192109755237;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
                        result[0] += -0.07311443311304443;
                      } else {
                        result[0] += -0.004082631766678752;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.022312657825461685;
            }
          }
        } else {
          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.011192850483560954;
              } else {
                result[0] += -0.03407394952547548;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                  result[0] += 0.010179832345893414;
                } else {
                  result[0] += -0.004762453695836475;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                  result[0] += -0.08199814001358482;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.060294389724732333) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                        result[0] += -0.07136915171913769;
                      } else {
                        if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                          result[0] += 0.08723391267332906;
                        } else {
                          result[0] += -0.09966863251469117;
                        }
                      }
                    } else {
                      result[0] += -0.009696445976919871;
                    }
                  } else {
                    result[0] += 0.0823017707819018;
                  }
                }
              }
            }
          } else {
            result[0] += -0.04662182263200082;
          }
        }
      }
    } else {
      result[0] += 0.0003049864232300039;
    }
  } else {
    result[0] += 0.11263710602407069;
  }
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
    if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.03420138359069913) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
              result[0] += -0.008808004293657964;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.0038683044989849934;
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.12369472450094937;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                            result[0] += 0.023098295812719903;
                          } else {
                            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.921924352645874468) ) ) {
                              result[0] += -0.044687395583639794;
                            } else {
                              result[0] += 0.07650317342464631;
                            }
                          }
                        } else {
                          result[0] += 0.09784723042011939;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                        result[0] += 0.026554407823765853;
                      } else {
                        result[0] += 0.00015513586303031443;
                      }
                    }
                  }
                } else {
                  result[0] += 0.016737589409466617;
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.868834793567657693) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                        result[0] += 0.00741701007680118;
                      } else {
                        result[0] += -0.02443335435365872;
                      }
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.725620865821838823) ) ) {
                        result[0] += -0.0015207359580225056;
                      } else {
                        result[0] += 0.022704536916101768;
                      }
                    }
                  } else {
                    result[0] += 0.015838504396524428;
                  }
                } else {
                  result[0] += 0.06524644238887226;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.881510615348816362) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.017014344346302616;
                  } else {
                    result[0] += -0.03626388875242622;
                  }
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.0009447597059853612;
                  } else {
                    result[0] += -0.03535402421153994;
                  }
                }
              } else {
                result[0] += 0.017350691425237295;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                result[0] += -0.0004878765800511469;
              } else {
                result[0] += 0.028569113759055548;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
              if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.0009108515547493227;
              } else {
                result[0] += 0.00888921409274706;
              }
            } else {
              result[0] += -0.0009991448520437925;
            }
          } else {
            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.0007987058827446423;
            } else {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.06896924972534357) ) ) {
                  result[0] += -0.01943800209270391;
                } else {
                  if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += 0.012217025334144406;
                  } else {
                    result[0] += -0.011244425959730058;
                  }
                }
              } else {
                result[0] += 6.424338834916197e-05;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
          result[0] += -0.028807745860858366;
        } else {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.95751476287841975) ) ) {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.477602720260621005) ) ) {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.581332206726075107) ) ) {
                    if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.417592287063599077) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.026417016983033115) ) ) {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                            result[0] += 0.008230630749084691;
                          } else {
                            result[0] += -0.10493776811427191;
                          }
                        } else {
                          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.013334269209854414;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
                              result[0] += 0.09098886389106119;
                            } else {
                              result[0] += 0.027329955516167483;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.06360033375548096;
                      }
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
                          result[0] += -0.04435557490677033;
                        } else {
                          result[0] += 0.013762974407665694;
                        }
                      } else {
                        result[0] += -0.019426072215074707;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                      result[0] += -0.032492689600534884;
                    } else {
                      if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.01028845829141481;
                      } else {
                        result[0] += 0.013469533178828914;
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.825982809066773349) ) ) {
                        result[0] += -9.795340886029805e-05;
                      } else {
                        result[0] += -0.024007058121182447;
                      }
                    } else {
                      result[0] += 0.004290874030292386;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                      result[0] += 0.017088167973325154;
                    } else {
                      result[0] += -0.01581319588154622;
                    }
                  }
                }
              } else {
                result[0] += -0.012724534369573177;
              }
            } else {
              if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.724856853485109198) ) ) {
                    result[0] += 0.003019470478122688;
                  } else {
                    result[0] += 0.02423989491978573;
                  }
                } else {
                  result[0] += -0.0013830889126903063;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                      result[0] += 0.08670109340536176;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += -0.007866339399686477;
                      } else {
                        result[0] += 0.03282374784280801;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                        result[0] += -0.0873280133812071;
                      } else {
                        result[0] += -0.012500417810121862;
                      }
                    } else {
                      result[0] += -0.046764030453421555;
                    }
                  }
                } else {
                  result[0] += 0.0011205975785412764;
                }
              }
            }
          } else {
            result[0] += -0.014772182618358188;
          }
        }
      }
    } else {
      result[0] += 0.0002985675831391067;
    }
  } else {
    result[0] += 0.11263710602407069;
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
        result[0] += 0.00021823081427209588;
      } else {
        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.012675821781158891) ) ) {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.04393117438582723;
              } else {
                result[0] += 0.014655284137989022;
              }
            } else {
              result[0] += 0.07028884210004567;
            }
          } else {
            if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
                            result[0] += 0.010342715084847796;
                          } else {
                            result[0] += -0.1617033819729384;
                          }
                        } else {
                          result[0] += 0.031058352388125512;
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                          result[0] += 0.027068555230447062;
                        } else {
                          result[0] += -0.013660328162015232;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += 0.00422976311042201;
                        } else {
                          result[0] += -0.06725717432423425;
                        }
                      } else {
                        if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                          result[0] += -0.08037740574170855;
                        } else {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                            result[0] += -0.030019467266193818;
                          } else {
                            result[0] += 0.13943258837033076;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      result[0] += -0.07687022829939173;
                    } else {
                      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.08572211713116162;
                      } else {
                        result[0] += 0.011148492532193811;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.97887301445007413) ) ) {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.706861495971680576) ) ) {
                      result[0] += 0.006904440497463167;
                    } else {
                      result[0] += 0.03710240067878325;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.620046615600586826) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                        result[0] += 0.0357889351231081;
                      } else {
                        if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                          result[0] += -0.014453673350107322;
                        } else {
                          result[0] += 0.004244925014990865;
                        }
                      }
                    } else {
                      result[0] += 0.019277941414087087;
                    }
                  }
                }
              } else {
                result[0] += -0.0003638624006143869;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += -0.04170446729563059;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.7800335884094256) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.551017761230469638) ) ) {
                            result[0] += -0.03268993150990985;
                          } else {
                            result[0] += 0.07579378830699308;
                          }
                        } else {
                          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.32411074638366788) ) ) {
                              result[0] += 0.058225499180430185;
                            } else {
                              result[0] += -0.03862085810044372;
                            }
                          } else {
                            result[0] += -0.01474605068454729;
                          }
                        }
                      } else {
                        result[0] += 0.002417555798870749;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += 0.05448337616692631;
                      } else {
                        result[0] += -0.0523458333379954;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.025518756269124962;
                    } else {
                      result[0] += 0.015203814103069722;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.006620267305977367;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
                      result[0] += 0.01872007875811636;
                    } else {
                      result[0] += 0.050782453740748704;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.645421981811524326) ) ) {
                        result[0] += 0.025518296351413464;
                      } else {
                        result[0] += -0.005623075438246131;
                      }
                    } else {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.863673448562622958) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += -0.10738234462997323;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                            result[0] += 0.0263046766736863;
                          } else {
                            result[0] += -0.007224748387977049;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.613121509552002841) ) ) {
                          result[0] += -0.008564100572036402;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.576439857482911933) ) ) {
                            result[0] += -0.04580278911472339;
                          } else {
                            result[0] += -0.003421338371744591;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                          result[0] += 0.00036158983669479266;
                        } else {
                          result[0] += 0.016789639997891192;
                        }
                      } else {
                        if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
                          if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                              result[0] += 0.014041527889274825;
                            } else {
                              result[0] += -0.04179126512467909;
                            }
                          } else {
                            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                                result[0] += -0.07160754801135329;
                              } else {
                                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                  result[0] += 0.02831895958538921;
                                } else {
                                  result[0] += -0.019223896067991617;
                                }
                              }
                            } else {
                              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                                result[0] += -0.10233716239939242;
                              } else {
                                result[0] += -0.03005077102985937;
                              }
                            }
                          }
                        } else {
                          result[0] += 0.008719618566900353;
                        }
                      }
                    } else {
                      result[0] += -0.12283590063429146;
                    }
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.970040798187256748) ) ) {
            result[0] += -0.0133295375394481;
          } else {
            result[0] += 0.09557385955186269;
          }
        }
      }
    } else {
      result[0] += 0.11116080367092865;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
      result[0] += 0.006246023258910621;
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.156774044036865678) ) ) {
        result[0] += 0.021764541986821262;
      } else {
        result[0] += 0.09395922134553707;
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
      if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
            if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.0008247981672163959;
                  } else {
                    result[0] += -0.09105772289985398;
                  }
                } else {
                  result[0] += -0.07681257031799335;
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.913499355316162998) ) ) {
                  result[0] += -0.06191260836130446;
                } else {
                  result[0] += -0.27284757827185474;
                }
              }
            } else {
              result[0] += 0.11043926060462;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
              result[0] += 0.008316184834528478;
            } else {
              result[0] += 0.049487787074829634;
            }
          }
        } else {
          result[0] += 0.09619993313593617;
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[20].missing != -1) || (data[20].fvalue <= (double)48.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += -0.005073184899150609;
              } else {
                if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)8816427008.000001907) ) ) {
                  result[0] += 0.0402861010436598;
                } else {
                  result[0] += -0.05528357434453553;
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.004144246045392759;
              } else {
                if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.84983205795288264) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                      result[0] += 0.011444258127931496;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.006135184884300074;
                        } else {
                          result[0] += -0.026212781474965332;
                        }
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += -0.02659097378238881;
                        } else {
                          result[0] += -0.00034896049757562224;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.06534191000058828;
                    } else {
                      result[0] += -0.006096386459422882;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.589327573776246005) ) ) {
                      result[0] += -0.008449804589399135;
                    } else {
                      result[0] += 0.0253209335118248;
                    }
                  } else {
                    result[0] += 0.0013181493079498113;
                  }
                }
              }
            }
          } else {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.768316030502320224) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  if ( LIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                        result[0] += 0.031306914568499995;
                      } else {
                        result[0] += -0.006678606167047627;
                      }
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.725620865821838823) ) ) {
                        result[0] += -0.03072220191409449;
                      } else {
                        result[0] += 0.0053152460736981015;
                      }
                    }
                  } else {
                    result[0] += -0.12737207697338537;
                  }
                } else {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.0026902194382244808;
                      } else {
                        result[0] += -0.10076494831601426;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
                        result[0] += 0.05235431832929032;
                      } else {
                        result[0] += 0.013345819372695363;
                      }
                    }
                  } else {
                    result[0] += -0.005821892140464191;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03247094127549038;
                  } else {
                    if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                        result[0] += 0.1129134976528896;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.59600305557251154) ) ) {
                          result[0] += -0.012995386640764273;
                        } else {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                            result[0] += 0.07171538775275908;
                          } else {
                            result[0] += 0.0017592416933896423;
                          }
                        }
                      }
                    } else {
                      result[0] += -0.005091464407380483;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
                      result[0] += -0.05243519658640127;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)7.970394372940064365) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.026807529720914887;
                        } else {
                          result[0] += -0.05166485134790825;
                        }
                      } else {
                        result[0] += 0.204344626496631;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                        if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.18467869052136357;
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
                              result[0] += -0.08369269437208116;
                            } else {
                              result[0] += 0.03717812333056227;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
                              if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                                  result[0] += -0.011400408933196098;
                                } else {
                                  result[0] += 0.18248131872464857;
                                }
                              } else {
                                result[0] += -0.004759975647340207;
                              }
                            } else {
                              result[0] += -0.022155176289489716;
                            }
                          } else {
                            result[0] += -0.09038425380450865;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += 0.09312017511210297;
                        } else {
                          result[0] += -0.005867071149347838;
                        }
                      }
                    } else {
                      result[0] += -0.04572517075197158;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  result[0] += -0.016616183772442755;
                } else {
                  result[0] += 0.023835023691404195;
                }
              } else {
                result[0] += -0.028828072183263376;
              }
            }
          }
        } else {
          result[0] += 0.0001239969286897892;
        }
      }
    } else {
      result[0] += 0.11116080367092865;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
      result[0] += 0.006198332081893117;
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
        result[0] += 0.02163802623069855;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
          result[0] += 0.18899725203234247;
        } else {
          result[0] += 0.04436939874104076;
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.417592287063599077) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.901921629905701128) ) ) {
              result[0] += 0.1552679538600211;
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.442198753356934482) ) ) {
                result[0] += -0.005099957056885056;
              } else {
                result[0] += 0.10515414195343196;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
              result[0] += -0.027214180672890914;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                result[0] += -0.05719326789593866;
              } else {
                result[0] += -0.23898308405172727;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
            result[0] += 0.008966940387067355;
          } else {
            result[0] += 0.049530867046630386;
          }
        }
      } else {
        if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
              if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += -0.00471034182266961;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.09408408207861041;
                } else {
                  if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)12.00000000000000178) ) ) {
                    result[0] += -0.12166238178008759;
                  } else {
                    result[0] += -1.8232419935616853e-05;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)2.500000000000000444) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += -0.06207835513214571;
                  } else {
                    result[0] += -0.0022264032693797915;
                  }
                } else {
                  result[0] += 0.007222518689832726;
                }
              } else {
                if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                  result[0] += 0.0005717775698109336;
                } else {
                  result[0] += -0.04891058378105952;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
              if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += 0.006870670077240092;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += -0.002465504156176893;
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    result[0] += -0.008013095498029144;
                  } else {
                    result[0] += -0.03135467891140403;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                    result[0] += -0.03326009593439844;
                  } else {
                    result[0] += 0.1576139560523735;
                  }
                } else {
                  if ( UNLIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.05683391249781447;
                  } else {
                    result[0] += 0.011670680828741287;
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.802901029586792436) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                    if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.11250272492296216;
                      } else {
                        result[0] += -0.007801433757423941;
                      }
                    } else {
                      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.109050035476685458) ) ) {
                            result[0] += -0.03872176451586403;
                          } else {
                            result[0] += 0.08419376876281098;
                          }
                        } else {
                          result[0] += 0.04891879711238349;
                        }
                      } else {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += -0.07055759924466637;
                              } else {
                                result[0] += -0.01988761400635349;
                              }
                            } else {
                              if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)24.00000000000000355) ) ) {
                                result[0] += -0.030275523542379747;
                              } else {
                                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.549068689346314365) ) ) {
                                  result[0] += 0.008680274365581745;
                                } else {
                                  result[0] += 0.1137801242007378;
                                }
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                                result[0] += -0.03383791668670408;
                              } else {
                                result[0] += 0.09868007084449126;
                              }
                            } else {
                              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)240.0000000000000284) ) ) {
                                      result[0] += 0.03716155044184283;
                                    } else {
                                      result[0] += -0.09968111842214586;
                                    }
                                  } else {
                                    result[0] += -0.07092934924470408;
                                  }
                                } else {
                                  result[0] += -0.11032391511044788;
                                }
                              } else {
                                result[0] += 0.03363048215817698;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.026417016983033115) ) ) {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.740319490432739702) ) ) {
                                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.700190305709839755) ) ) {
                                  result[0] += -0.021770759976875557;
                                } else {
                                  result[0] += 0.10835395011145586;
                                }
                              } else {
                                result[0] += -0.05743456102568528;
                              }
                            } else {
                              result[0] += 0.08010929418903007;
                            }
                          } else {
                            result[0] += -0.14068746909424465;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.016618836126989094;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                          result[0] += 0.1728231502474249;
                        } else {
                          result[0] += 0.016939508198608042;
                        }
                      } else {
                        result[0] += -0.006111234584420095;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0028517296032620023;
                }
              }
            }
          }
        } else {
          result[0] += 0.00010670994881355019;
        }
      }
    } else {
      result[0] += 0.11116080367092865;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          result[0] += 0.017759347806606676;
        } else {
          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
            result[0] += 0.16548594743454484;
          } else {
            result[0] += 0.0176238611985212;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)7.134879350662232333) ) ) {
          result[0] += -0.00015680663105848157;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.183107137680054599) ) ) {
            result[0] += 0.11547342528183019;
          } else {
            result[0] += 0.013431051807599653;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
        result[0] += 0.021474970025368852;
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.339936256408692294) ) ) {
          result[0] += 0.025788354004098214;
        } else {
          result[0] += 0.16102676188183118;
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)1.242453336715698464) ) ) {
        if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
          if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)6.218359947204590732) ) ) {
                result[0] += -0.006923315713736073;
              } else {
                result[0] += -0.06349996200754704;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.12252235412597834) ) ) {
                result[0] += -0.0151095455914429;
              } else {
                result[0] += -0.18573097951920658;
              }
            }
          } else {
            result[0] += 0.10721775715518043;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.895678043365479404) ) ) {
            result[0] += 0.008523395885754683;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
              result[0] += 0.09996412460258955;
            } else {
              result[0] += 0.03286183802595188;
            }
          }
        }
      } else {
        result[0] += 0.09476630543209236;
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
            result[0] += 0.0008507524308286442;
          } else {
            if ( LIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.00017107345253670808;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                        result[0] += -0.012476441552003791;
                      } else {
                        if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.02134592361152563;
                        } else {
                          result[0] += -0.054868494256409286;
                        }
                      }
                    } else {
                      result[0] += -0.04625844706270907;
                    }
                  } else {
                    if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
                        result[0] += -0.009618714391572997;
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.025167483769760475;
                        } else {
                          result[0] += -0.026748469874620092;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                        result[0] += -0.08090365514241027;
                      } else {
                        result[0] += -0.019752776195426244;
                      }
                    }
                  }
                } else {
                  result[0] += -3.27840274854624e-05;
                }
              }
            } else {
              result[0] += -0.00042421280748028157;
            }
          }
        } else {
          result[0] += 0.0001053371569504147;
        }
      } else {
        if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.27480554580688654) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.09085798263549982) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.951058745384216753) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.615975379943848544) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.349750161170959917) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.481121778488159624) ) ) {
                          result[0] += -0.017797644245058346;
                        } else {
                          result[0] += 0.05879783507393549;
                        }
                      } else {
                        result[0] += -0.06635653552050141;
                      }
                    } else {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.21245336532592951) ) ) {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.552491664886475498) ) ) {
                          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                              result[0] += -0.11958448452287479;
                            } else {
                              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
                                result[0] += 0.014505999537141406;
                              } else {
                                result[0] += -0.03468807584347318;
                              }
                            }
                          } else {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                              result[0] += -0.09671597225747759;
                            } else {
                              result[0] += -0.020925294650905795;
                            }
                          }
                        } else {
                          result[0] += -0.13683010708874574;
                        }
                      } else {
                        result[0] += -0.18524206607598945;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += 0.016515378258334098;
                    } else {
                      result[0] += 0.11729439949436507;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.553655147552491123) ) ) {
                    result[0] += 0.17642317848107192;
                  } else {
                    if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.11697985114755693;
                      } else {
                        result[0] += 0.008629601227003264;
                      }
                    } else {
                      result[0] += 0.0904618223950211;
                    }
                  }
                }
              } else {
                result[0] += -0.13353337032404222;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.537947177886963779) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
                  result[0] += -0.10775236787872162;
                } else {
                  result[0] += 0.020047923653966244;
                }
              } else {
                result[0] += -0.13762908569841598;
              }
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += -0.05885842020216763;
                  } else {
                    result[0] += 0.11367006774150785;
                  }
                } else {
                  result[0] += 0.1344103008331998;
                }
              } else {
                result[0] += -0.045809052241784165;
              }
            } else {
              result[0] += 0.0735679301809058;
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.1375558201423298;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                  result[0] += 0.08149654464873003;
                } else {
                  result[0] += -0.0734092256053848;
                }
              }
            } else {
              result[0] += 0.2858563734881659;
            }
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
              result[0] += -0.18063215793627285;
            } else {
              result[0] += -0.026589440630312697;
            }
          }
        }
      }
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          result[0] += 0.017482761977811942;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
            result[0] += 0.16905234257741486;
          } else {
            result[0] += 0.02004176114420567;
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)7.134879350662232333) ) ) {
          result[0] += -0.00019859337586342017;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.183107137680054599) ) ) {
            result[0] += 0.11548293114452923;
          } else {
            result[0] += 0.014088464414406782;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
        result[0] += 0.021209044887406225;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
          result[0] += 0.19021154311388563;
        } else {
          result[0] += 0.04182816363445491;
        }
      }
    }
  }
  if ( LIKELY( !(data[42].missing != -1) || (data[42].fvalue <= (double)4.500000000000000888) ) ) {
    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)8.43217372894287287) ) ) {
      if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
        result[0] += -0.00010059599770617066;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                    result[0] += -0.019443792975510257;
                  } else {
                    result[0] += -0.058695108725084724;
                  }
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.007691145784908784;
                  } else {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.010174952796097151;
                    } else {
                      result[0] += -0.12448041405438628;
                    }
                  }
                }
              } else {
                result[0] += 0.023061642720396153;
              }
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                    result[0] += -0.0012324169797783954;
                  } else {
                    result[0] += 0.06543366074210724;
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                    result[0] += 0.0803208404160831;
                  } else {
                    result[0] += -0.08799899534550237;
                  }
                }
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
                  result[0] += -0.0021451630355082954;
                } else {
                  result[0] += -0.06000751012258006;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.568724632263184482) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                          result[0] += -0.046435197767733256;
                        } else {
                          result[0] += 0.08147252549452115;
                        }
                      } else {
                        result[0] += -0.04442200353994541;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                        result[0] += 0.030069450667972238;
                      } else {
                        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.726826429367066318) ) ) {
                          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                            result[0] += -0.00463973138409582;
                          } else {
                            result[0] += -0.0856437994062556;
                          }
                        } else {
                          result[0] += 0.025433241965182464;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                      result[0] += 0.10545131803733379;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.42478513717651456) ) ) {
                        result[0] += 0.018824066565025083;
                      } else {
                        result[0] += -0.022542115922654885;
                      }
                    }
                  }
                } else {
                  result[0] += -0.0114337640957139;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)7.944137096405030185) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.00506416376299258;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.551017761230469638) ) ) {
                      result[0] += 0.024106145353705734;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.249904870986938921) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)8.08096981048584162) ) ) {
                          result[0] += 0.012189237016572763;
                        } else {
                          result[0] += 0.082163621325676;
                        }
                      } else {
                        result[0] += -0.012631305870915224;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.05792233570662947;
                  } else {
                    result[0] += -0.0725415003620279;
                  }
                }
              }
            } else {
              result[0] += 0.016384088942967726;
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                result[0] += 0.0006276975824938453;
              } else {
                result[0] += -0.034467103288787646;
              }
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += 0.012900002473074143;
                } else {
                  result[0] += 0.08142823820648018;
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.122815132141115058) ) ) {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
                    result[0] += 0.012351776419993353;
                  } else {
                    result[0] += 0.08086026848449356;
                  }
                } else {
                  result[0] += -0.004831280270832958;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.970257759094240058) ) ) {
                result[0] += -0.07351035347959971;
              } else {
                result[0] += 0.03676598873516692;
              }
            } else {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.03595960970189379;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                        result[0] += 0.0658144813276144;
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.013205529304327824;
                        } else {
                          result[0] += -0.019424759963649663;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                          result[0] += 0.014959876737841014;
                        } else {
                          result[0] += -0.02535740821381842;
                        }
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += 0.06500329959605401;
                        } else {
                          result[0] += -0.04197528349775344;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.044042788080693174;
                }
              } else {
                if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.005313006987368054;
                  } else {
                    result[0] += 0.06575214463027457;
                  }
                } else {
                  result[0] += -0.01113882773859168;
                }
              }
            }
          }
        }
      }
    } else {
      result[0] += 0.11116080367092865;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.863673448562622958) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
          result[0] += 0.017281158214073066;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
            result[0] += 0.1701671921111032;
          } else {
            result[0] += 0.01972780655305301;
          }
        }
      } else {
        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
          result[0] += -0.00026589152665152456;
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.183107137680054599) ) ) {
            result[0] += 0.11352823839577933;
          } else {
            result[0] += 0.015424038475738526;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
        result[0] += 0.021014615982436226;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
          result[0] += 0.19072715715745653;
        } else {
          result[0] += 0.04082255539418401;
        }
      }
    }
  }
  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
    result[0] += 0.000149010297774744;
  } else {
    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.262283086776734287) ) ) {
        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
          result[0] += 0.009603262060909785;
        } else {
          result[0] += 0.0004317999307820312;
        }
      } else {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.10313596840537731;
        } else {
          result[0] += -0.01792965251949012;
        }
      }
    } else {
      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.183107137680054599) ) ) {
          if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              result[0] += 0.007400498596362744;
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
                result[0] += -0.01253388363132395;
              } else {
                result[0] += 0.06235013663276301;
              }
            }
          } else {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += -0.004343874827879204;
                } else {
                  result[0] += -0.09145632667223126;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.018311096498961326;
                } else {
                  if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.0016471699534142726;
                  } else {
                    result[0] += 0.137724566630091;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.46012759208679288) ) ) {
                  result[0] += -0.051931162565758915;
                } else {
                  result[0] += 0.011341307903953601;
                }
              } else {
                result[0] += 0.005292877486030661;
              }
            }
          }
        } else {
          result[0] += -0.029332517169013323;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
              result[0] += -0.009142739568463908;
            } else {
              result[0] += 0.0053753614077601455;
            }
          } else {
            result[0] += -0.004939456333680998;
          }
        } else {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += -0.0005393048353646983;
                } else {
                  result[0] += -0.017250864113557966;
                }
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += -0.02658618757032415;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)80.00000000000001421) ) ) {
                        if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.013705314364827598;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.892816066741945136) ) ) {
                            result[0] += -0.049783051532709906;
                          } else {
                            result[0] += 0.08830696384628765;
                          }
                        }
                      } else {
                        result[0] += -0.03070047170451558;
                      }
                    } else {
                      if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                          result[0] += 0.01719397953878015;
                        } else {
                          result[0] += -0.029972874674379126;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                          if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                            result[0] += 0.010983314607650608;
                          } else {
                            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                                result[0] += -0.06176211842736204;
                              } else {
                                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                                  result[0] += -0.048741803269026884;
                                } else {
                                  result[0] += 0.06286263478823535;
                                }
                              }
                            } else {
                              result[0] += 0.07478864222578939;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                              if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                                result[0] += -0.011669001596702394;
                              } else {
                                result[0] += -0.13459535655510932;
                              }
                            } else {
                              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                                    result[0] += 0.0036629895709976087;
                                  } else {
                                    result[0] += 0.08964979902390652;
                                  }
                                } else {
                                  result[0] += 0.09684779209779755;
                                }
                              } else {
                                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)48.00000000000000711) ) ) {
                                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                                    result[0] += -0.09876481166929;
                                  } else {
                                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.870983839035034624) ) ) {
                                      result[0] += -0.023932717118800326;
                                    } else {
                                      result[0] += 0.056893542758730015;
                                    }
                                  }
                                } else {
                                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                                    result[0] += -0.009894438127553619;
                                  } else {
                                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)40.00000000000000711) ) ) {
                                      result[0] += -0.0009779516460711342;
                                    } else {
                                      result[0] += 0.0897125151350282;
                                    }
                                  }
                                }
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.645421981811524326) ) ) {
                              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.66412305831909357) ) ) {
                                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.650573849678039995) ) ) {
                                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                                    result[0] += -0.004887331279411984;
                                  } else {
                                    result[0] += 0.04450847445102577;
                                  }
                                } else {
                                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.861792564392090288) ) ) {
                                    result[0] += -0.022588360927086507;
                                  } else {
                                    result[0] += -0.15624094179688927;
                                  }
                                }
                              } else {
                                result[0] += 0.021409624694485742;
                              }
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.777633190155030185) ) ) {
                                result[0] += 0.08435362727775425;
                              } else {
                                result[0] += 0.020600842474371814;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                    result[0] += -0.06288025445660327;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.58491539955139249) ) ) {
                      result[0] += -0.026254460446702266;
                    } else {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)10.87822628021240412) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                          result[0] += -0.004698069380258101;
                        } else {
                          if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                            result[0] += 0.018183997385223253;
                          } else {
                            result[0] += 0.09295959562455548;
                          }
                        }
                      } else {
                        result[0] += -0.0910641812177478;
                      }
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                result[0] += -0.001278891345815035;
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += 0.03910363627620954;
                } else {
                  result[0] += -0.015827426282649378;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.249904870986938921) ) ) {
              result[0] += -0.0008468105165572711;
            } else {
              result[0] += 0.022954950502851516;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      result[0] += -0.00034293747994142257;
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
            if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
              result[0] += -0.0148393474793043;
            } else {
              result[0] += 0.010194022831409681;
            }
          } else {
            if ( UNLIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                result[0] += 0.002285602854602445;
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.658699750900269443) ) ) {
                      result[0] += -0.03263195692458527;
                    } else {
                      result[0] += 0.005784464333353843;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.13002538681030451) ) ) {
                      result[0] += -0.045883145218039594;
                    } else {
                      if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                        result[0] += -0.03880774457476325;
                      } else {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += 0.08578423301985914;
                        } else {
                          result[0] += -0.01235463587444289;
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.051484699177772035;
                }
              }
            } else {
              result[0] += -0.001007316995871011;
            }
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    result[0] += 0.0003204626155110442;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                      result[0] += 0.0029511530540846796;
                    } else {
                      result[0] += 0.025451787599337163;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += -0.0032886836219016705;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        result[0] += -0.036490071187932634;
                      } else {
                        result[0] += 0.07586865608833675;
                      }
                    } else {
                      result[0] += 0.010740144280054218;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                  result[0] += -0.03798766741971396;
                } else {
                  if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.13839721679687678) ) ) {
                      result[0] += -0.0013811478833710457;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.616744756698609287) ) ) {
                        result[0] += 0.02689151297523686;
                      } else {
                        result[0] += -0.022684731574729573;
                      }
                    }
                  } else {
                    result[0] += -0.009348563589595836;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.124553249100018;
              } else {
                if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.02226504498291943;
                  } else {
                    result[0] += -0.15196951313023885;
                  }
                } else {
                  result[0] += 0.001761370430621476;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
              if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
                result[0] += -0.07124553774950826;
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                      result[0] += 0.05952898291647311;
                    } else {
                      if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                        result[0] += 0.0278040467040202;
                      } else {
                        result[0] += -0.008372705618188208;
                      }
                    }
                  } else {
                    result[0] += -0.02882642250183484;
                  }
                } else {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.025045459554581224;
                  } else {
                    result[0] += 0.04739377805096623;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                    result[0] += -0.0016521370919658205;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                      result[0] += -0.0010620145273247123;
                    } else {
                      result[0] += 0.027119862438205307;
                    }
                  }
                } else {
                  result[0] += -0.005248242367081629;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.184114694595337802) ) ) {
                  result[0] += 0.0035337826019348744;
                } else {
                  result[0] += 0.014431637189820563;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
          if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
              result[0] += 0.04056082285408499;
            } else {
              result[0] += -0.00010676019713406668;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
              if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.08098486680524089;
              } else {
                result[0] += -0.029618456715121495;
              }
            } else {
              if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.11407232284546076) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += -0.007848655883825701;
                    } else {
                      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                          result[0] += 0.028346553005730676;
                        } else {
                          result[0] += -0.05670610072790625;
                        }
                      } else {
                        result[0] += 0.05762625325217981;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                      result[0] += -0.015134235972679311;
                    } else {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                        result[0] += 0.03934369845318546;
                      } else {
                        result[0] += 0.004001040635018093;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY(  (data[43].missing != -1) && (data[43].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                    result[0] += 0.030571429090166607;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.02818941983464724;
                    } else {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.695914268493653232) ) ) {
                        result[0] += 0.03889054898243629;
                      } else {
                        result[0] += -0.027608451781934513;
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                  result[0] += -0.004597832090087291;
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                    result[0] += -0.06593154025222456;
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                      result[0] += -0.06784558068311426;
                    } else {
                      result[0] += 0.013169909827709778;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.002527763398833127;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      result[0] += -0.00047108924230556667;
    } else {
      result[0] += -0.0140450906861059;
    }
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
      if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
        result[0] += -0.0002414496799792202;
      } else {
        if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.66339445114135831) ) ) {
              result[0] += 0.010912569450746194;
            } else {
              if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)3072.000000000000455) ) ) {
                result[0] += -0.0800672967315489;
              } else {
                result[0] += -0.02162804909973361;
              }
            }
          } else {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.400584220886231357) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.32411074638366788) ) ) {
                result[0] += -0.0011003163540283512;
              } else {
                result[0] += -0.3349935113780715;
              }
            } else {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)7.286531209945679599) ) ) {
                result[0] += 0.04465887501897143;
              } else {
                result[0] += -0.1310281446615634;
              }
            }
          }
        } else {
          result[0] += -0.09669466077910215;
        }
      }
    } else {
      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.740565299987793857) ) ) {
            result[0] += -0.00114767503557085;
          } else {
            result[0] += -0.014308847048783727;
          }
        } else {
          if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)21466447872.00000381) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                  if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                    result[0] += 0.0002971754389927267;
                  } else {
                    result[0] += 0.013070788571526546;
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
                    result[0] += -0.0030707267943761605;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        result[0] += -0.03618297064177247;
                      } else {
                        result[0] += 0.06882161320845293;
                      }
                    } else {
                      result[0] += 0.009427759662174526;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
                    result[0] += -0.027864402847800368;
                  } else {
                    result[0] += 0.0013329461837996773;
                  }
                } else {
                  result[0] += -0.00907366562452048;
                }
              }
            } else {
              result[0] += -0.021644014045349165;
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.311129093170167792) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.933422565460205966) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.010773086769331775;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                      result[0] += 0.02036598829620132;
                    } else {
                      result[0] += -0.007712133491184343;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
                    result[0] += -0.0358971699444746;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
                      result[0] += -0.025510771328551746;
                    } else {
                      result[0] += 0.01962227576180076;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.00016406010194347314;
                  } else {
                    result[0] += 0.010960321044099197;
                  }
                } else {
                  result[0] += 0.013057561939691004;
                }
              }
            } else {
              if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)6.000000000000000888) ) ) {
                result[0] += -0.03337460903295621;
              } else {
                result[0] += 0.04099048220108126;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.9236645698547381) ) ) {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)176.0000000000000284) ) ) {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.265274047851563388) ) ) {
                if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                      if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
                        result[0] += 0.07694414596604982;
                      } else {
                        result[0] += 0.016480242018002658;
                      }
                    } else {
                      result[0] += -0.01795799419284988;
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.5708808898925799) ) ) {
                      result[0] += -0.09254344986622093;
                    } else {
                      result[0] += 0.10551024758908722;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                    result[0] += 0.016997075891813915;
                  } else {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
                      result[0] += -0.0065472056940837605;
                    } else {
                      result[0] += -0.05369933417491597;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.479143142700197089) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.645740747451783115) ) ) {
                    if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                      result[0] += 0.01604309754814124;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.388237953186036044) ) ) {
                        result[0] += -0.018253950123439593;
                      } else {
                        result[0] += -0.06590558050567381;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                      result[0] += -0.0043633597884364436;
                    } else {
                      result[0] += 0.022177330920565428;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.241249561309815341) ) ) {
                    result[0] += 0.02095989497416219;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.700753688812257636) ) ) {
                      result[0] += 0.052595456558387293;
                    } else {
                      result[0] += -0.008160033820138158;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += -0.027625888588480862;
              } else {
                result[0] += 0.004445257409015728;
              }
            }
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.731793165206910068) ) ) {
              result[0] += -0.02883635435664952;
            } else {
              result[0] += 0.01019251200468678;
            }
          }
        } else {
          result[0] += 0.0022051141086701703;
        }
      }
    }
  } else {
    if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
      result[0] += -0.0004464475933475026;
    } else {
      if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.449861526489258257) ) ) {
          result[0] += -0.012763101280984762;
        } else {
          result[0] += -0.03866519223994855;
        }
      } else {
        if ( LIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)6.000000000000000888) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42895507812500178) ) ) {
              result[0] += 0.13439815025451848;
            } else {
              result[0] += 0.015487908723296024;
            }
          } else {
            result[0] += -0.014270989456608235;
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.303973913192749912) ) ) {
            result[0] += -0.028660536369323072;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.81278371810913264) ) ) {
              result[0] += -0.00045665873839228967;
            } else {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.009921303741023346;
              } else {
                result[0] += 0.09549861481208918;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)3.000000000000000444) ) ) {
    if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
      if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
          result[0] += 0.007388621524349934;
        } else {
          if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
            if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += 0.012007904802841328;
                } else {
                  result[0] += -0.09709364684927944;
                }
              } else {
                result[0] += -0.04739219062783064;
              }
            } else {
              result[0] += -0.01181537773165653;
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.35061407089233576) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.257122993469240058) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.025192260742188388) ) ) {
                  result[0] += -0.042423301031603466;
                } else {
                  if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)7.500000000000000888) ) ) {
                    result[0] += -0.00298839785205641;
                  } else {
                    result[0] += 0.034402205875790785;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.005041088712762884;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.239300251007080966) ) ) {
                    result[0] += 0.04055085145571003;
                  } else {
                    result[0] += -0.060752958755305814;
                  }
                }
              }
            } else {
              result[0] += 0.062410565789534925;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
          if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)8.500000000000001776) ) ) {
            if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.42478513717651456) ) ) {
              if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.966960191726685458) ) ) {
                  result[0] += 0.0034530461202266108;
                } else {
                  result[0] += -0.023760320003406818;
                }
              } else {
                if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.655387401580811435) ) ) {
                    result[0] += 0.013449497680791634;
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.07032398365587642;
                    } else {
                      result[0] += 0.011565972187225973;
                    }
                  }
                } else {
                  result[0] += -0.005524999480290387;
                }
              }
            } else {
              if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                  result[0] += -0.039135261708499756;
                } else {
                  result[0] += -5.131383813140631e-06;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
                  result[0] += -0.007895977176469804;
                } else {
                  result[0] += -0.051829050811028746;
                }
              }
            }
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                  if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.847910165786744052) ) ) {
                    result[0] += 0.03333760020018713;
                  } else {
                    result[0] += -0.01138894942560047;
                  }
                } else {
                  result[0] += -0.005859916982220464;
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.726826429367066318) ) ) {
                  result[0] += -0.011789528646996436;
                } else {
                  if ( UNLIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += -0.028898752893829968;
                  } else {
                    if ( UNLIKELY( !(data[48].missing != -1) || (data[48].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.551017761230469638) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.568724632263184482) ) ) {
                          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                            result[0] += -0.029060856557996385;
                          } else {
                            result[0] += 0.10102298855659063;
                          }
                        } else {
                          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.138696432113648349) ) ) {
                            if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.248264408048404;
                            } else {
                              result[0] += -0.009621023263077847;
                            }
                          } else {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += -0.0035442372020431444;
                            } else {
                              result[0] += -0.12231662223503728;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.01255395617211051;
                      }
                    } else {
                      result[0] += 0.002333919644651877;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.913499355316162998) ) ) {
                  result[0] += -0.01748229117107633;
                } else {
                  result[0] += -0.12404121886258977;
                }
              } else {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)8.680339336395265448) ) ) {
                  result[0] += 0.007074809660169619;
                } else {
                  result[0] += 0.1752323747086748;
                }
              }
            }
          }
        } else {
          result[0] += -0.02879649139516124;
        }
      }
    } else {
      if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)11.50000000000000178) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)1.700598716735840066) ) ) {
          if ( LIKELY( !(data[37].missing != -1) || (data[37].fvalue <= (double)1.500000000000000222) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2352.000000000000455) ) ) {
                result[0] += -0.022535401221366685;
              } else {
                result[0] += 0.04697986971870262;
              }
            } else {
              result[0] += -0.024254873103834212;
            }
          } else {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)4.053106546401978427) ) ) {
              result[0] += -0.006441411521455523;
            } else {
              result[0] += 0.08712004866636687;
            }
          }
        } else {
          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.861792564392090288) ) ) {
            if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
              result[0] += 0.03610658367287981;
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.01634240150451749) ) ) {
                    result[0] += 0.00595601415929602;
                  } else {
                    result[0] += 0.136931349456017;
                  }
                } else {
                  result[0] += 0.07698902435635134;
                }
              } else {
                result[0] += -0.002308197312015598;
              }
            }
          } else {
            if ( UNLIKELY( !(data[38].missing != -1) || (data[38].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)48.00000000000000711) ) ) {
                    result[0] += -0.0115721260303602;
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)62.00000000000000711) ) ) {
                      result[0] += -0.0004757025546504637;
                    } else {
                      result[0] += 0.023136163822048184;
                    }
                  }
                } else {
                  result[0] += -0.007169328469979978;
                }
              } else {
                if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
                  if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.384246587753296343) ) ) {
                    result[0] += -0.014665500985326494;
                  } else {
                    result[0] += -0.06469912832324781;
                  }
                } else {
                  if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                    if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.005989327225119758;
                    } else {
                      result[0] += 0.055591832990828184;
                    }
                  } else {
                    result[0] += -0.01606753216799524;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)12.00000000000000178) ) ) {
                result[0] += -0.05119767239151293;
              } else {
                result[0] += -0.013723908282791254;
              }
            }
          }
        }
      } else {
        result[0] += 0.01961053774389429;
      }
    }
  } else {
    result[0] += 4.773145797218391e-05;
  }
  if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)3.500000000000000444) ) ) {
    result[0] += 9.254294708544206e-05;
  } else {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
              if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)240.0000000000000284) ) ) {
                result[0] += -0.03948284577712305;
              } else {
                result[0] += 0.10417642177683184;
              }
            } else {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.003031115304157618;
                } else {
                  result[0] += 0.031681080964391026;
                }
              } else {
                if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.041306169877332;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                    result[0] += 0.04349897123671272;
                  } else {
                    if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.009282686869292218;
                    } else {
                      result[0] += 0.03173195564898938;
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
              if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    result[0] += 0.06146151108888043;
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.138333082199097124) ) ) {
                      result[0] += -0.07827750772736958;
                    } else {
                      result[0] += 0.020592473586137805;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                          result[0] += -0.08865086243371739;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                            result[0] += -0.09238132445171346;
                          } else {
                            result[0] += 0.03384144178385253;
                          }
                        }
                      } else {
                        result[0] += 0.010220594526068234;
                      }
                    } else {
                      result[0] += 0.06198628481271695;
                    }
                  } else {
                    result[0] += -0.10057070950927692;
                  }
                }
              } else {
                result[0] += 0.011016348928272952;
              }
            } else {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82155513763427912) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
                    result[0] += -0.032497631512280134;
                  } else {
                    if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)7.500000000000000888) ) ) {
                      result[0] += 0.013267018753838454;
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.036670446395874912) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.01634240150451749) ) ) {
                            result[0] += -0.1261891426413632;
                          } else {
                            result[0] += -0.015957122672314698;
                          }
                        } else {
                          result[0] += 0.018068826750638433;
                        }
                      } else {
                        result[0] += -0.04517385601686724;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
                    result[0] += -0.015829297578622413;
                  } else {
                    if ( UNLIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      result[0] += 0.001415563640668344;
                    } else {
                      result[0] += 0.0626188836378482;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.06632852554321467) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.011601869118796845;
                  } else {
                    result[0] += 0.03597156391762424;
                  }
                } else {
                  result[0] += -0.02822360019409292;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
              result[0] += 0.004962806115509542;
            } else {
              result[0] += -0.017621245194753946;
            }
          } else {
            if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              result[0] += 0.0038172486061972344;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                result[0] += -0.024971813812578902;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                    result[0] += -0.02499965722774656;
                  } else {
                    if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.57691621780395685) ) ) {
                        result[0] += 0.035509432464772984;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                          result[0] += -0.013886822065397444;
                        } else {
                          result[0] += -0.22582640629184136;
                        }
                      }
                    } else {
                      result[0] += 0.05982197817085658;
                    }
                  }
                } else {
                  result[0] += -0.0016640201627485729;
                }
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)48.00000000000000711) ) ) {
          if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2565.000000000000455) ) ) {
            result[0] += -0.0005320257298588353;
          } else {
            result[0] += 0.01710147779278799;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                  result[0] += 0.011253265795434987;
                } else {
                  result[0] += 0.04170336933152927;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                  result[0] += -0.03296389098399683;
                } else {
                  result[0] += 0.007154073241364568;
                }
              }
            } else {
              result[0] += -0.008991762203131819;
            }
          } else {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)3.500000000000000444) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
                if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2727.500000000000455) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.008030065750541894;
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
                        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
                          result[0] += -0.024091947247530467;
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                            result[0] += 0.0060614783597437456;
                          } else {
                            result[0] += -0.013695120205638265;
                          }
                        }
                      } else {
                        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                          result[0] += -0.0037528219478142424;
                        } else {
                          result[0] += 0.031230307414329597;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.266057968139650214) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.960975408554078037) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                            result[0] += -0.04071800781209228;
                          } else {
                            result[0] += 0.028596572933488835;
                          }
                        } else {
                          result[0] += -0.003717649741041702;
                        }
                      } else {
                        result[0] += -0.008484126223108325;
                      }
                    }
                  }
                } else {
                  result[0] += -0.03796643598547105;
                }
              } else {
                result[0] += -0.07937768205038859;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.184114694595337802) ) ) {
                result[0] += -0.015783977368962688;
              } else {
                result[0] += 0.028121089346540534;
              }
            }
          }
        }
      }
    } else {
      result[0] += -0.012586193012324984;
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -9.56088447064932e-05;
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.138696432113648349) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.497866153717041238) ) ) {
              result[0] += -0.003276399961173286;
            } else {
              result[0] += -0.14081546262282713;
            }
          } else {
            result[0] += -0.031792139959818465;
          }
        } else {
          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)5.519456863403321201) ) ) {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.0001185708453766002;
              } else {
                result[0] += -0.05113012345334936;
              }
            } else {
              result[0] += 0.07657074663603625;
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
                result[0] += 0.01651442538526851;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                  result[0] += 0.002538586783627509;
                } else {
                  result[0] += 0.0804315550143853;
                }
              }
            } else {
              result[0] += -0.03122831199619666;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)3.067782521247864214) ) ) {
            if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += -0.008624121334376065;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.960975408554078037) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.825982809066773349) ) ) {
                            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                              result[0] += -0.011348877522827669;
                            } else {
                              result[0] += 0.031137544390480516;
                            }
                          } else {
                            result[0] += 0.06132326109907446;
                          }
                        } else {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.08407373665289823;
                          } else {
                            result[0] += 0.011531041617928159;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
                          result[0] += 0.05037122500319574;
                        } else {
                          result[0] += 0.020951329133207203;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += 0.03628028243123417;
                    } else {
                      result[0] += -0.0026014191275930882;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.043585707625689044;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.42478513717651456) ) ) {
                      result[0] += 0.036134423135851405;
                    } else {
                      result[0] += -0.1271512931783578;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.248013019561768466) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.339936256408692294) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                        result[0] += 0.01999430984907629;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                          result[0] += 0.009615145226277296;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.874179124832154208) ) ) {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += -0.08173079415254719;
                            } else {
                              result[0] += 0.004060923025218254;
                            }
                          } else {
                            result[0] += -0.011434849703068375;
                          }
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.10740698612741024;
                      } else {
                        result[0] += -0.026596920092119863;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
                      result[0] += 0.06862268277749868;
                    } else {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.052717685699463779) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.706861495971680576) ) ) {
                            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                              result[0] += 0.019528346494005863;
                            } else {
                              result[0] += -0.05164350960444606;
                            }
                          } else {
                            result[0] += 0.07369102523239981;
                          }
                        } else {
                          result[0] += -0.006316171158024695;
                        }
                      } else {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.384246587753296343) ) ) {
                          result[0] += -0.008243050169421213;
                        } else {
                          result[0] += -0.09496972397130536;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.339535951614380771) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                        result[0] += -0.07890693416021778;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.278613805770874912) ) ) {
                          result[0] += 0.005306041563862688;
                        } else {
                          result[0] += -0.06257337567372638;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.052717685699463779) ) ) {
                          result[0] += 0.04420107478550347;
                        } else {
                          result[0] += 0.11976771466191666;
                        }
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.01205021871010339;
                        } else {
                          result[0] += 0.07371034615285205;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.03792068621346365;
                  }
                }
              }
            } else {
              result[0] += 0.038388841377818075;
            }
          } else {
            result[0] += -0.02711815390628707;
          }
        } else {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
            result[0] += 0.007345078869973116;
          } else {
            result[0] += -0.015294495016265742;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
              result[0] += 0.014514473845017143;
            } else {
              result[0] += -0.0397240081781193;
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
              if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                  result[0] += 0.10933020639464913;
                } else {
                  result[0] += -0.025003208337270995;
                }
              } else {
                result[0] += -0.038400966076934835;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.288254261016846591) ) ) {
                result[0] += -0.02733799500631024;
              } else {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.422362327575684482) ) ) {
                    result[0] += 0.05023146297314716;
                  } else {
                    result[0] += -0.036174400994839505;
                  }
                } else {
                  result[0] += -0.0017018659527589922;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.013257823923561658;
          } else {
            result[0] += -0.0400536557312063;
          }
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)8.053540229797365058) ) ) {
          result[0] += 0.0007213190349328424;
        } else {
          result[0] += 0.06156635037818138;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00010387048651395151;
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.67577242851257413) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.138333082199097124) ) ) {
            result[0] += 0.021896438561407005;
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += -0.019397009200053557;
              } else {
                if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.88192772865295499) ) ) {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.041053720398744026;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.729812622070313388) ) ) {
                          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.108135223388672763) ) ) {
                            result[0] += 0.020725723373203414;
                          } else {
                            result[0] += -0.06908614634891383;
                          }
                        } else {
                          result[0] += 0.08658540436148321;
                        }
                      }
                    } else {
                      result[0] += 0.0649146728460981;
                    }
                  } else {
                    result[0] += -0.011305655437904107;
                  }
                } else {
                  result[0] += 0.07767538192116188;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                  result[0] += -0.03473589840528173;
                } else {
                  result[0] += -0.12482397163160544;
                }
              } else {
                result[0] += 0.02939301881920103;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
            result[0] += -0.0363855111091397;
          } else {
            if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.379319667816162998) ) ) {
              result[0] += 0.014431228279720444;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.0892077726981387;
                } else {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
                      result[0] += -0.09534096257352488;
                    } else {
                      result[0] += 0.061742732300676666;
                    }
                  } else {
                    result[0] += -0.03158060363753445;
                  }
                }
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
                    result[0] += 0.008767131543692356;
                  } else {
                    result[0] += -0.02425122511580144;
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)6.237347126007080966) ) ) {
                    result[0] += 0.03268779524474266;
                  } else {
                    result[0] += 0.21136968034653825;
                  }
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
            result[0] += 0.010242196357203207;
          } else {
            result[0] += 0.09442977547433809;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.006677311820934695;
              } else {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.932935476303101474) ) ) {
                  result[0] += 0.04964508568716694;
                } else {
                  result[0] += 0.13595424982737345;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.745216369628907138) ) ) {
                  result[0] += -0.0451160237417077;
                } else {
                  result[0] += 0.007393729103314229;
                }
              } else {
                result[0] += 0.04656081260032877;
              }
            }
          } else {
            result[0] += -0.010165933208141211;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.02556846082095418;
        } else {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.0766609723405579;
          } else {
            result[0] += -0.0034723229046628072;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.86655306816101163) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                    result[0] += -0.014359378166013112;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.917705297470093662) ) ) {
                        result[0] += 0.11472847607276432;
                      } else {
                        result[0] += 0.020980576584906114;
                      }
                    } else {
                      result[0] += 0.01169746273902462;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.036670446395874912) ) ) {
                    result[0] += 0.000987322295757098;
                  } else {
                    result[0] += -0.04117418245706697;
                  }
                }
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.487163543701172763) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.605120182037354404) ) ) {
                          result[0] += -0.05102305794460015;
                        } else {
                          result[0] += 0.019525888746081913;
                        }
                      } else {
                        result[0] += -0.0750744607413268;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.109050035476685458) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.052717685699463779) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.497866153717041238) ) ) {
                            result[0] += 0.043618447236369505;
                          } else {
                            result[0] += -0.01739947272205599;
                          }
                        } else {
                          result[0] += -0.12654298545102202;
                        }
                      } else {
                        result[0] += 0.024315793800738374;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                      result[0] += -0.11480803716547397;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.092434883117676669) ) ) {
                        result[0] += 0.03984577823609345;
                      } else {
                        result[0] += -0.08579545856221682;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                      result[0] += -0.04920238400584215;
                    } else {
                      result[0] += 0.033275185192935444;
                    }
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                        result[0] += 0.08822553176416341;
                      } else {
                        result[0] += 0.020094573270979483;
                      }
                    } else {
                      result[0] += -0.008855800523381675;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.05033273616150118;
            }
          } else {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.531409263610840732) ) ) {
              result[0] += 0.014359294515176896;
            } else {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                result[0] += 0.11156949583910075;
              } else {
                result[0] += 0.028660290015053325;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
              result[0] += -0.05423790067944989;
            } else {
              result[0] += 0.008559656623833142;
            }
          } else {
            result[0] += -0.015277628032063631;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00011258613219441435;
  } else {
    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.860215187072755683) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                    result[0] += 0.05331076897826377;
                  } else {
                    result[0] += -0.02478499024583258;
                  }
                } else {
                  result[0] += -0.06392066984042892;
                }
              } else {
                result[0] += 0.05109034456377503;
              }
            } else {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.274755001068116123) ) ) {
                  result[0] += 0.0331242081932387;
                } else {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                    result[0] += 0.002040568064286146;
                  } else {
                    result[0] += -0.03486727166349723;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.120943069458008701) ) ) {
                  result[0] += -0.059115245281353174;
                } else {
                  result[0] += -0.01756851857167429;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.211602926254273349) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.602003335952759233) ) ) {
                      result[0] += 0.026459728001118434;
                    } else {
                      result[0] += -0.03788230242660204;
                    }
                  } else {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                      result[0] += 0.055396252610639546;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                          result[0] += -0.04480164560929657;
                        } else {
                          result[0] += 0.0759821589804071;
                        }
                      } else {
                        result[0] += -0.04483895091921007;
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.02976703379526252;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                      result[0] += 0.018733635865897855;
                    } else {
                      result[0] += 0.0598532667356767;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.11326837539672896) ) ) {
                    result[0] += 0.07003589975595897;
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                      result[0] += 0.0326054713762577;
                    } else {
                      result[0] += -0.018176494744993162;
                    }
                  }
                } else {
                  result[0] += 0.07678898130058907;
                }
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.02329502278093372;
                  } else {
                    result[0] += 0.038718567320971085;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.238486170768738237) ) ) {
                      result[0] += -0.06759000886747937;
                    } else {
                      result[0] += 0.007286029706826978;
                    }
                  } else {
                    if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.11326837539672896) ) ) {
                      result[0] += -0.002789944824659739;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.666320323944092685) ) ) {
                        result[0] += 0.10692516581265131;
                      } else {
                        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.597218394279480425) ) ) {
                            result[0] += -0.03401551915106855;
                          } else {
                            result[0] += 0.03952699814631992;
                          }
                        } else {
                          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.068990230560303623) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
                              result[0] += 0.025428746485243427;
                            } else {
                              if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.921924352645874468) ) ) {
                                result[0] += 0.1512822411130861;
                              } else {
                                result[0] += 0.059614149533090834;
                              }
                            }
                          } else {
                            result[0] += 0.010859410055506969;
                          }
                        }
                      }
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                  result[0] += -0.049289382046144;
                } else {
                  result[0] += 0.0006497718678795346;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
            result[0] += -0.0399326732094212;
          } else {
            if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.436733961105347568) ) ) {
              result[0] += -0.017694446667336976;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += -0.026678369474223165;
              } else {
                result[0] += 0.018032982145348837;
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
            result[0] += 0.02211086571547989;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
              result[0] += -0.07184416467208093;
            } else {
              result[0] += -0.03174470136673201;
            }
          }
        } else {
          if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)2.012675821781158891) ) ) {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.740319490432739702) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.868834793567657693) ) ) {
                result[0] += 0.0624289941923066;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                  result[0] += 0.0741025438268784;
                } else {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.44140100479126021) ) ) {
                    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
                      result[0] += -0.0160834270115922;
                    } else {
                      result[0] += -0.1107795309868645;
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                        result[0] += -0.0070784063098318405;
                      } else {
                        result[0] += -0.05608971665057821;
                      }
                    } else {
                      result[0] += 0.002581327623079904;
                    }
                  }
                }
              }
            } else {
              result[0] += -0.0406140205097985;
            }
          } else {
            result[0] += 0.10498231360152188;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.464467763900757724) ) ) {
        if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += 0.003520437246031675;
        } else {
          result[0] += 0.047451620373921116;
        }
      } else {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
            if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += 0.019146038971590663;
              } else {
                result[0] += -0.046918616742384324;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                result[0] += -0.010444477536681408;
              } else {
                result[0] += 0.08711314022439832;
              }
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.641084194183350498) ) ) {
              result[0] += -0.009248909668130753;
            } else {
              result[0] += -0.0718785515782387;
            }
          }
        } else {
          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.624251961708069292) ) ) {
            result[0] += 0.027482601798482012;
          } else {
            result[0] += -0.057569926702848356;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00012902375003823215;
  } else {
    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.420525312423706943) ) ) {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.138696432113648349) ) ) {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
            result[0] += -0.03608219740973985;
          } else {
            result[0] += 0.00364561789282154;
          }
        } else {
          if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += -0.020924698316034416;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.040618419647218573) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += -0.006897874692258514;
              } else {
                result[0] += 0.0762193948129948;
              }
            } else {
              result[0] += -0.05605881532766624;
            }
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.817651987075806108) ) ) {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.701225757598877397) ) ) {
            if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.003838300704956943) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.624251961708069292) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += 0.046100824156591974;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                            result[0] += 0.049906112437523714;
                          } else {
                            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                              result[0] += 0.018429494638726562;
                            } else {
                              result[0] += -0.017767048783592584;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                              result[0] += 0.021401022926298238;
                            } else {
                              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                                result[0] += 0.039372039714158535;
                              } else {
                                result[0] += 0.1002361904623059;
                              }
                            }
                          } else {
                            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.540854334831238237) ) ) {
                                result[0] += 0.016845629987145234;
                              } else {
                                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.238486170768738237) ) ) {
                                  result[0] += -0.11276570776247051;
                                } else {
                                  result[0] += -0.0011515356241783839;
                                }
                              }
                            } else {
                              result[0] += -0.03605076061537904;
                            }
                          }
                        }
                      }
                    } else {
                      result[0] += -0.050882029160476475;
                    }
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                      result[0] += -0.010516175034427767;
                    } else {
                      result[0] += -0.06079690305564936;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
                    result[0] += 0.09216786379762684;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.597218394279480425) ) ) {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                        result[0] += 0.017352663301669843;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.566809177398682529) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.012675821781158891) ) ) {
                            result[0] += -0.04018967677867865;
                          } else {
                            result[0] += 0.03442082143314117;
                          }
                        } else {
                          result[0] += 0.015310166001968687;
                        }
                      }
                    } else {
                      result[0] += 0.012404528419272719;
                    }
                  }
                }
              } else {
                result[0] += -0.08065847450635732;
              }
            } else {
              result[0] += 0.0733220224777147;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.087577104568482333) ) ) {
              result[0] += 0.07342726694105886;
            } else {
              result[0] += 0.018646765423996913;
            }
          }
        } else {
          result[0] += -0.0025577027428815716;
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.700598716735840066) ) ) {
          result[0] += -0.023901216112253793;
        } else {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
            result[0] += 0.08115986829690808;
          } else {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.624251961708069292) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.823630809783937323) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += -0.05351583886699623;
                } else {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                      result[0] += -0.06180009857464726;
                    } else {
                      result[0] += 0.038073286570702976;
                    }
                  } else {
                    result[0] += -0.08142798517200744;
                  }
                }
              } else {
                result[0] += 0.02219043777460028;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += -0.06808437208142087;
                  } else {
                    result[0] += 0.02187126886025443;
                  }
                } else {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.005311555264230397;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.66339445114135831) ) ) {
                      result[0] += 0.11438615772832571;
                    } else {
                      result[0] += 0.037923229232826455;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                    result[0] += -0.11368088481005884;
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
                      result[0] += 0.1302938180270318;
                    } else {
                      result[0] += 0.02618575787082942;
                    }
                  }
                } else {
                  result[0] += -0.01457091950890882;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
          if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)8.053540229797365058) ) ) {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.00665250006673914;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.597218394279480425) ) ) {
                  result[0] += -0.03574413781386811;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += 0.042695121783047456;
                  } else {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.874179124832154208) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
                        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += 0.05632853311204566;
                        } else {
                          result[0] += -0.026105510845768884;
                        }
                      } else {
                        result[0] += -0.07708162401334712;
                      }
                    } else {
                      result[0] += 0.031036744674625322;
                    }
                  }
                }
              }
            } else {
              result[0] += 0.08620588295510896;
            }
          } else {
            result[0] += -0.008234131641986856;
          }
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
              result[0] += -0.05507090745065303;
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.723882198333742011) ) ) {
                result[0] += 0.017388248518157286;
              } else {
                result[0] += -0.006289315306051886;
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.837713479995728427) ) ) {
                result[0] += 0.022967175589215863;
              } else {
                result[0] += -0.025216064629620245;
              }
            } else {
              result[0] += -0.018511475155908053;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00012731821447384617;
  } else {
    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.982408046722412998) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.744781017303467685) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.934867382049561435) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
              if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                result[0] += 0.011524180881778482;
              } else {
                result[0] += 0.07624549653843414;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                result[0] += 0.019585234976672503;
              } else {
                result[0] += -0.07105815563137118;
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.524927973747253862) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                  result[0] += 0.061157601562776144;
                } else {
                  result[0] += -0.011105337701840218;
                }
              } else {
                result[0] += -0.04183153013250717;
              }
            } else {
              result[0] += -0.0005992967995884926;
            }
          }
        } else {
          result[0] += 0.08171429484265928;
        }
      } else {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                  result[0] += 0.08429683387800382;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += -0.006776474956392806;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.051854133605957919) ) ) {
                      result[0] += 0.0316889891485711;
                    } else {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.067782521247864214) ) ) {
                        result[0] += 0.020203355602598085;
                      } else {
                        result[0] += -0.019509306446216693;
                      }
                    }
                  }
                }
              } else {
                result[0] += -0.004586558062763488;
              }
            } else {
              result[0] += -0.01116021626853117;
            }
          } else {
            if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.935600519180298074) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                    result[0] += -0.014521492063738936;
                  } else {
                    result[0] += -0.12466997611650216;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.06865505975580057;
                    } else {
                      result[0] += 0.02680390334580689;
                    }
                  } else {
                    result[0] += 0.005056935342433228;
                  }
                }
              } else {
                result[0] += 0.015296855517752615;
              }
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)0.8958797454833985485) ) ) {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.138696432113648349) ) ) {
                        result[0] += 0.09577424161766608;
                      } else {
                        result[0] += 0.004775187619580457;
                      }
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                        result[0] += -0.09018911474666387;
                      } else {
                        result[0] += -0.004551129940166078;
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.022538185119630683) ) ) {
                      result[0] += 0.014150351087150952;
                    } else {
                      result[0] += 0.07868127579147331;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                    result[0] += 0.06820876000198114;
                  } else {
                    result[0] += 0.018063733967063244;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.400584220886231357) ) ) {
                  result[0] += 0.07819700581893191;
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.875080585479737216) ) ) {
                      result[0] += -0.04246993809184671;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        result[0] += -0.007062879995815359;
                      } else {
                        if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.036670446395874912) ) ) {
                            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.020127415657043901) ) ) {
                              result[0] += -0.01387527324918257;
                            } else {
                              result[0] += 0.09790865728438387;
                            }
                          } else {
                            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.729812622070313388) ) ) {
                              result[0] += -0.05530941831988128;
                            } else {
                              result[0] += 0.035888691945248796;
                            }
                          }
                        } else {
                          result[0] += 0.09356918322633816;
                        }
                      }
                    }
                  } else {
                    result[0] += -0.104411852765731;
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.04930099555924082;
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.000938536164761554;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += 0.01686891785245529;
            } else {
              if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.068990230560303623) ) ) {
                    result[0] += 0.08351852372704845;
                  } else {
                    result[0] += -0.005183297167213256;
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += -0.0502998784645331;
                  } else {
                    result[0] += 0.04295457909055642;
                  }
                }
              } else {
                result[0] += 0.13977347355902922;
              }
            }
          } else {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.11326837539672896) ) ) {
                result[0] += 0.048747450743994836;
              } else {
                if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                  result[0] += -0.043717155709201844;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42895507812500178) ) ) {
                    result[0] += 0.009740345142953171;
                  } else {
                    result[0] += -0.03972590857557722;
                  }
                }
              }
            } else {
              result[0] += 0.02914323799361651;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.138333082199097124) ) ) {
          result[0] += -0.0648591403806816;
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                result[0] += 0.11085589476769145;
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.556798219680787021) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += -0.01714553957966086;
                  } else {
                    result[0] += 0.025548764693990242;
                  }
                } else {
                  result[0] += -0.037187950203756676;
                }
              }
            } else {
              result[0] += -0.0053452447538649495;
            }
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
              result[0] += 0.11937885843921431;
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.025049086915235327;
                } else {
                  result[0] += 0.07556161187566424;
                }
              } else {
                result[0] += 0.011653549822783209;
              }
            }
          }
        }
      }
    }
  }
  if ( UNLIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)1.00000001800250948e-35) ) ) {
    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
        if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
          if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)96.00000000000001421) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                result[0] += 0.011409991874099263;
              } else {
                result[0] += 0.037769248488872115;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.901921629905701128) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += 0.001260128743048789;
                  } else {
                    result[0] += 0.0961135280576326;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.182021141052246982) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.35526132583618342) ) ) {
                      result[0] += -0.046438543424417555;
                    } else {
                      result[0] += -0.11841448581055142;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.018134998795204086;
                      } else {
                        result[0] += 0.10876159853504377;
                      }
                    } else {
                      result[0] += -0.0352390150003135;
                    }
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                  result[0] += -0.0218338708775235;
                } else {
                  result[0] += 0.0167443501392763;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += 0.023327908853020177;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.73867654800415217) ) ) {
                  result[0] += 0.014200884769920276;
                } else {
                  if ( LIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)17137926144.00000191) ) ) {
                    result[0] += -0.09028597566994548;
                  } else {
                    result[0] += 0.014955203830670089;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.85429954528808771) ) ) {
                result[0] += -0.007741646878851885;
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.497866153717041238) ) ) {
                  result[0] += -0.12256268198372954;
                } else {
                  result[0] += -0.023374486502943083;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.537837505340577948) ) ) {
            if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.589234352111818183) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)4.166635274887085849) ) ) {
                  if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.380914688110353339) ) ) {
                      result[0] += -0.03843212245577058;
                    } else {
                      result[0] += -0.14062140000732778;
                    }
                  } else {
                    if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2415.000000000000455) ) ) {
                      if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
                        result[0] += -0.020570318835906883;
                      } else {
                        result[0] += -0.07983822536391126;
                      }
                    } else {
                      result[0] += -0.006556835603435957;
                    }
                  }
                } else {
                  result[0] += 0.08708813882593769;
                }
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                  result[0] += -0.02153411348980995;
                } else {
                  if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
                    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += -0.027232874188429526;
                    } else {
                      result[0] += 0.035934388075823456;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.025192260742188388) ) ) {
                      result[0] += -0.029456303782689888;
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.018466510580364633;
                      } else {
                        result[0] += 0.05905504141486389;
                      }
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
                result[0] += 0.009438454114935936;
              } else {
                result[0] += -0.008276087307276634;
              }
            }
          } else {
            if ( LIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)3.500000000000000444) ) ) {
              result[0] += -0.0016727315518881754;
            } else {
              if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += 0.01148261945491357;
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.58491539955139249) ) ) {
                    result[0] += 0.030902970744100186;
                  } else {
                    result[0] += 0.10313612763757468;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.20763492584228693) ) ) {
                    result[0] += -0.004467387113240293;
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
                      result[0] += -0.022947172705194484;
                    } else {
                      if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                        result[0] += 0.2111255355422298;
                      } else {
                        result[0] += 0.06164453644509413;
                      }
                    }
                  }
                } else {
                  result[0] += -0.03296294539188912;
                }
              }
            }
          }
        }
      } else {
        result[0] += 0.0009137462601779809;
      }
    } else {
      if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)3.500000000000000444) ) ) {
        if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)6.000000000000000888) ) ) {
          result[0] += -0.03875665633953082;
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.252218484878540927) ) ) {
              result[0] += 0.035389944737163614;
            } else {
              result[0] += -0.060984401862821974;
            }
          } else {
            if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)160.0000000000000284) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                result[0] += -0.025571792372562997;
              } else {
                result[0] += -0.12303182413866393;
              }
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
                result[0] += -0.10112086787273532;
              } else {
                result[0] += 0.051072611721510314;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)56.00000000000000711) ) ) {
            result[0] += -0.05310525839778274;
          } else {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                result[0] += -0.0059457355286934195;
              } else {
                result[0] += 0.06692355884370921;
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.12411325154985053;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                  if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += -0.02359939457158669;
                  } else {
                    result[0] += 0.03391269080377376;
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.966329097747803623) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.481121778488159624) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.82617378234863459) ) ) {
                        if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.349750161170959917) ) ) {
                          result[0] += -0.11851248731798142;
                        } else {
                          result[0] += 0.015665639084874675;
                        }
                      } else {
                        result[0] += -0.07689080923504846;
                      }
                    } else {
                      result[0] += -0.058369589270878486;
                    }
                  } else {
                    result[0] += -0.00137472084213774;
                  }
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.092434883117676669) ) ) {
            result[0] += -0.04330449253521633;
          } else {
            result[0] += -0.13630766311715184;
          }
        }
      }
    }
  } else {
    result[0] += 0.0001905836950411232;
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -7.436072181750393e-05;
  } else {
    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                  result[0] += -0.05155039149658841;
                } else {
                  result[0] += 0.011538304131107412;
                }
              } else {
                result[0] += 0.03375569074007609;
              }
            } else {
              result[0] += -0.03412602064561548;
            }
          } else {
            result[0] += -0.013590092333259016;
          }
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.176905632019043857) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.211602926254273349) ) ) {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                  result[0] += 0.06411935177762422;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.729812622070313388) ) ) {
                    result[0] += -0.021788112506513278;
                  } else {
                    result[0] += 0.021794045880896725;
                  }
                }
              } else {
                if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                  result[0] += 0.0051538954784664465;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.178976058959961826) ) ) {
                    result[0] += 0.016271279971291446;
                  } else {
                    result[0] += -0.08784401428778262;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.521452903747559482) ) ) {
                  result[0] += 0.03194256121423552;
                } else {
                  result[0] += 0.08311502989252584;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.248013019561768466) ) ) {
                  result[0] += -0.0683266377053088;
                } else {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01856290302991281;
                  } else {
                    result[0] += 0.0511581451237583;
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.08030730313267141;
            } else {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.178966760635376865) ) ) {
                result[0] += -0.052396307063912996;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.684611082077027255) ) ) {
                    result[0] += 0.0527319296099852;
                  } else {
                    result[0] += -0.03100952199318723;
                  }
                } else {
                  result[0] += -0.02273311436271653;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.119004011154175693) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                  result[0] += -0.09773877510386386;
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.597218394279480425) ) ) {
                    result[0] += -0.053682155659938315;
                  } else {
                    result[0] += 0.06648061727093456;
                  }
                }
              } else {
                result[0] += 0.06413467944424418;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                result[0] += -0.09223925615621736;
              } else {
                result[0] += -0.0044440096271741265;
              }
            }
          } else {
            result[0] += -0.03174004170416205;
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.008349562811580554;
            } else {
              result[0] += 0.057200038913888585;
            }
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                result[0] += 0.04554839097545271;
              } else {
                result[0] += -0.035096773141292746;
              }
            } else {
              result[0] += 0.016471434444495597;
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.932935476303101474) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                  result[0] += -0.022792421467132423;
                } else {
                  result[0] += 0.025759900902669133;
                }
              } else {
                result[0] += -0.0254863533967171;
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)4.855921268463135654) ) ) {
                result[0] += -0.0016200954199269075;
              } else {
                result[0] += 0.057979915637354024;
              }
            }
          } else {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                    result[0] += 0.04140449197655773;
                  } else {
                    result[0] += -0.06287030545739423;
                  }
                } else {
                  result[0] += -0.048163200533595164;
                }
              } else {
                result[0] += -0.04616581399282116;
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
                result[0] += 0.008745864012894419;
              } else {
                result[0] += -0.04320065039496457;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)2.770631790161133257) ) ) {
            result[0] += 0.08921714255048374;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.36105370521545499) ) ) {
              result[0] += 0.0048805486547187195;
            } else {
              if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
                result[0] += -0.006812151775756722;
              } else {
                result[0] += 0.1374080786749888;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.0883984565734881) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
              if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
                  result[0] += 0.027656654656754227;
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                      result[0] += -0.0753253370661303;
                    } else {
                      result[0] += 0.02654335180466323;
                    }
                  } else {
                    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.087577104568482333) ) ) {
                          result[0] += -0.0877826704818145;
                        } else {
                          result[0] += -0.01131348758122104;
                        }
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
                          result[0] += 0.04649596129752087;
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                            result[0] += -0.0908569227467838;
                          } else {
                            result[0] += 0.007161330346577999;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.11479258167974227;
                    }
                  }
                }
              } else {
                result[0] += 0.07289866514335717;
              }
            } else {
              result[0] += -0.02716610970049703;
            }
          } else {
            result[0] += -0.01754184079802806;
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.0007037803018854869;
          } else {
            result[0] += 0.1878106182127357;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -9.852999374029326e-05;
  } else {
    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.597218394279480425) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.267844915390015537) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.060294389724732333) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                result[0] += -0.01279706704816552;
              } else {
                if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.582024335861206943) ) ) {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
                    result[0] += 0.03396257496055115;
                  } else {
                    if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                      result[0] += 0.010376000429024462;
                    } else {
                      result[0] += -0.07909370175089742;
                    }
                  }
                } else {
                  result[0] += 0.05529793715015633;
                }
              }
            } else {
              if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.11326837539672896) ) ) {
                result[0] += -0.008145747869555896;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
                  result[0] += -0.011349812002648303;
                } else {
                  result[0] += -0.09623573436185744;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.349460363388062412) ) ) {
              result[0] += 0.08509073825587911;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
                        result[0] += -0.05036045283355545;
                      } else {
                        result[0] += 0.02711716732193119;
                      }
                    } else {
                      if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.0021875152567094507;
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.539549827575684482) ) ) {
                          result[0] += 0.043969452994926744;
                        } else {
                          result[0] += -0.054551117330747835;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.540854334831238237) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.874179124832154208) ) ) {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.867504835128785068) ) ) {
                          if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.04823887236704186;
                          } else {
                            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                              result[0] += 0.04339604249869569;
                            } else {
                              result[0] += -0.044896727788731806;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.673553824424744096) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                              if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                                    result[0] += -0.05088868164129556;
                                  } else {
                                    result[0] += -0.13873741753333188;
                                  }
                                } else {
                                  result[0] += -0.016408133637582912;
                                }
                              } else {
                                result[0] += -0.0017883034077126273;
                              }
                            } else {
                              result[0] += 0.015422883549285643;
                            }
                          } else {
                            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.378218650817871982) ) ) {
                                result[0] += -0.003552433709842856;
                              } else {
                                result[0] += -0.05176164308963789;
                              }
                            } else {
                              result[0] += 0.0235393804497857;
                            }
                          }
                        }
                      } else {
                        result[0] += 0.011221593050998211;
                      }
                    } else {
                      result[0] += 0.04807509915867344;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                    if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                      result[0] += -0.0006212428950482731;
                    } else {
                      result[0] += 0.030568540317385887;
                    }
                  } else {
                    result[0] += -0.015251937321121484;
                  }
                }
              } else {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.0019898361257276844;
                } else {
                  result[0] += -0.013925536564570304;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.731793165206910068) ) ) {
            result[0] += -0.0746404378830535;
          } else {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
              if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                result[0] += 0.018267919263802897;
              } else {
                result[0] += 0.09480668897454364;
              }
            } else {
              result[0] += 0.10236777951275305;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.499747991561890537) ) ) {
          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.303973913192749912) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.09618431162799547;
            } else {
              result[0] += -0.006049580177625051;
            }
          } else {
            result[0] += 0.010042708071802013;
          }
        } else {
          if ( LIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)96.00000000000001421) ) ) {
            if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
              if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.448499202728272373) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.382196187973023349) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                    result[0] += -0.014823139716619263;
                  } else {
                    result[0] += 0.015303677921857573;
                  }
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.037083305986549846;
                  } else {
                    result[0] += -0.012520411453178044;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                  result[0] += 0.15387736836612553;
                } else {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.021846807064144307;
                  } else {
                    result[0] += 0.02529870321903296;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.657235145568849433) ) ) {
                result[0] += -0.09000384739096852;
              } else {
                result[0] += 0.03314968194702463;
              }
            }
          } else {
            result[0] += 0.13609746620580357;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
        result[0] += 0.032999643852449655;
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.516392707824708808) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.350240230560303178) ) ) {
            if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0020632828146914224;
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.051747083663941318) ) ) {
                  result[0] += 0.02703260108310539;
                } else {
                  result[0] += 0.0855209356014574;
                }
              } else {
                if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.002764576692290943;
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.521452903747559482) ) ) {
                      result[0] += 0.14707486603214584;
                    } else {
                      result[0] += 0.028784171448674196;
                    }
                  }
                } else {
                  result[0] += -0.005362274490209321;
                }
              }
            }
          } else {
            result[0] += -0.025936582577547787;
          }
        } else {
          if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.08111345703226873;
          } else {
            if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.257356405258179155) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.27480554580688654) ) ) {
                result[0] += 0.026898952030673447;
              } else {
                result[0] += 0.1468302942713047;
              }
            } else {
              result[0] += -0.006024979098774332;
            }
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)24.00000000000000355) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.20086622238159357) ) ) {
        if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.86655306816101163) ) ) {
            result[0] += 0.003439028651359557;
          } else {
            result[0] += -0.02568506626418106;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.932935476303101474) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += -0.0026534470406119857;
            } else {
              result[0] += 0.009615822918953515;
            }
          } else {
            result[0] += -0.013874366514885576;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.861792564392090288) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
            result[0] += 0.009708288402676582;
          } else {
            result[0] += 0.06084814039965325;
          }
        } else {
          if ( LIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.04397060392737499;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                  result[0] += 0.004663759178200415;
                } else {
                  result[0] += 0.06548556052086367;
                }
              } else {
                if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  result[0] += -0.004814304464968731;
                } else {
                  result[0] += -0.023043848783551635;
                }
              }
            }
          } else {
            result[0] += -0.030045670659157073;
          }
        }
      }
    } else {
      result[0] += 0.00010151131135524039;
    }
  } else {
    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.921924352645874468) ) ) {
            if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.02201679354875915;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.043341875076294833) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)4.189540147781372958) ) ) {
                    result[0] += 0.05124555358814764;
                  } else {
                    result[0] += -0.03186578918458101;
                  }
                } else {
                  result[0] += 0.0433571398767568;
                }
              }
            } else {
              result[0] += -0.025942579082024006;
            }
          } else {
            result[0] += -0.013351319820697476;
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.587220668792725498) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.0361817411508212;
            } else {
              result[0] += 0.009006379704668515;
            }
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              result[0] += -0.07282198002838887;
            } else {
              if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.134879350662232333) ) ) {
                result[0] += 0.02705433176881071;
              } else {
                if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                  result[0] += -0.003922980232533973;
                } else {
                  result[0] += -0.09385605127220607;
                }
              }
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.581332206726075107) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.761470437049866167) ) ) {
            result[0] += -0.01706389670146646;
          } else {
            result[0] += 0.07866280604961663;
          }
        } else {
          if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.592359304428101474) ) ) {
            result[0] += 0.1247546765553608;
          } else {
            if ( UNLIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.780479431152345526) ) ) {
                  if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)7.338562726974488193) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.08234349146723986;
                    } else {
                      result[0] += 0.059306317385556785;
                    }
                  } else {
                    result[0] += 0.07098340606919887;
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.417800903320314276) ) ) {
                      result[0] += -0.09584536018334323;
                    } else {
                      result[0] += -0.017606921599776892;
                    }
                  } else {
                    result[0] += 0.07980654166104001;
                  }
                }
              } else {
                result[0] += -0.027194519124707828;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.060294389724732333) ) ) {
                  result[0] += 0.02326316575091703;
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                    result[0] += -0.08722667038157818;
                  } else {
                    result[0] += -0.021898265888321437;
                  }
                }
              } else {
                if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.388237953186036044) ) ) {
                  if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.190076351165773261) ) ) {
                    result[0] += -4.430902848725578e-05;
                  } else {
                    result[0] += -0.03745858791568141;
                  }
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.815665721893312323) ) ) {
                    result[0] += 0.10348313257350553;
                  } else {
                    result[0] += 0.016400446041585533;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)12.90173864364624201) ) ) {
          result[0] += 0.002713096802802607;
        } else {
          result[0] += -0.03772641825921109;
        }
      } else {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)12.0883984565734881) ) ) {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.09085798263549982) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.803987503051758701) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.511434078216553178) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                    result[0] += 0.03830769728334776;
                  } else {
                    result[0] += 0.11496205094836408;
                  }
                } else {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.01359428837042587;
                  } else {
                    result[0] += 0.06621307931471222;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.698346614837648261) ) ) {
                  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.825982809066773349) ) ) {
                      result[0] += -0.02840003123418469;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.791781663894654208) ) ) {
                        result[0] += 0.07484569385632098;
                      } else {
                        result[0] += -0.007399685060589145;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.04763970921636648;
                    } else {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.700598716735840066) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.169590950012207919) ) ) {
                          if ( UNLIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.02027363783937887;
                          } else {
                            result[0] += -0.025522140233234982;
                          }
                        } else {
                          result[0] += 0.05441126328633499;
                        }
                      } else {
                        result[0] += 0.044615652701708086;
                      }
                    }
                  }
                } else {
                  result[0] += -0.004573735545092319;
                }
              }
            } else {
              result[0] += -0.02612929070797886;
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
              result[0] += -0.08450010296705891;
            } else {
              result[0] += -0.005108412143010832;
            }
          }
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.005362363261705226;
          } else {
            result[0] += 0.16793518137395874;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00010278053375912761;
  } else {
    if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.804059982299805576) ) ) {
          if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.464467763900757724) ) ) {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.901921629905701128) ) ) {
                      result[0] += -0.05268160225381881;
                    } else {
                      result[0] += -0.012989283255250903;
                    }
                  } else {
                    result[0] += -0.08124321182040123;
                  }
                } else {
                  if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( LIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
                      result[0] += -0.014185847235158769;
                    } else {
                      result[0] += 0.09839454756662162;
                    }
                  } else {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.540854334831238237) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.418317794799805576) ) ) {
                        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.285735368728638583) ) ) {
                          if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.013959991604562611;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.198464870452881303) ) ) {
                              if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                result[0] += 0.08443309623566903;
                              } else {
                                result[0] += 0.016903138842829252;
                              }
                            } else {
                              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.569433569908142534) ) ) {
                                result[0] += -0.05597049860100332;
                              } else {
                                if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.645421981811524326) ) ) {
                                  result[0] += 0.07238757480010848;
                                } else {
                                  result[0] += -0.017295777201158134;
                                }
                              }
                            }
                          }
                        } else {
                          result[0] += -0.11324398180315279;
                        }
                      } else {
                        if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.531409263610840732) ) ) {
                          result[0] += 0.13810774961637273;
                        } else {
                          result[0] += 0.020722808835233954;
                        }
                      }
                    } else {
                      result[0] += -0.07172526573140374;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += 0.0010605192535691821;
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.138696432113648349) ) ) {
                    result[0] += 0.0754846542752735;
                  } else {
                    if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.382196187973023349) ) ) {
                        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.349750161170959917) ) ) {
                          result[0] += 0.029081189610672353;
                        } else {
                          result[0] += -0.09726533037410678;
                        }
                      } else {
                        result[0] += 0.10493629590475205;
                      }
                    } else {
                      result[0] += -0.02863713667848571;
                    }
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.717199802398682529) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.86655306816101163) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.003128293336473062;
                  } else {
                    result[0] += 0.07337285753079224;
                  }
                } else {
                  result[0] += 0.030793546742294804;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                  result[0] += 0.053695005180226865;
                } else {
                  result[0] += -0.02093938635500332;
                }
              }
            }
          } else {
            result[0] += 0.06558470517562148;
          }
        } else {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
              result[0] += -0.07013491806770931;
            } else {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.499747991561890537) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.726826429367066318) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                        result[0] += -0.1095400010408581;
                      } else {
                        result[0] += 0.0008455778021119631;
                      }
                    } else {
                      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
                        result[0] += 0.07657302757490186;
                      } else {
                        result[0] += -0.03769314763725146;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.122815132141115058) ) ) {
                      result[0] += 0.0248461258934579;
                    } else {
                      result[0] += -0.011914486324590993;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                    result[0] += 0.017555624056645684;
                  } else {
                    result[0] += 0.0940643319365273;
                  }
                }
              } else {
                if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.191013336181641069) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.01105935530087636;
                  } else {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.676220536231995073) ) ) {
                      result[0] += -0.0287168970024417;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        result[0] += -0.010636383638319535;
                      } else {
                        if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)6.23832273483276456) ) ) {
                          result[0] += 0.02520445639176815;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
                            result[0] += 0.12994192814756875;
                          } else {
                            result[0] += -0.033772277842680665;
                          }
                        }
                      }
                    }
                  }
                } else {
                  result[0] += -0.025287463059367684;
                }
              }
            }
          } else {
            result[0] += 0.05605888536050191;
          }
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.085941076278687412) ) ) {
          if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.768316030502320224) ) ) {
              result[0] += -0.005120075006286592;
            } else {
              result[0] += -0.07009034507835027;
            }
          } else {
            result[0] += -0.006714961910761239;
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.384246587753296343) ) ) {
            result[0] += 0.12715013983458165;
          } else {
            result[0] += -0.012137638873384251;
          }
        }
      }
    } else {
      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)6.239300251007080966) ) ) {
        if ( UNLIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
          result[0] += -0.0030920638196964465;
        } else {
          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.868834793567657693) ) ) {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)0.8958797454833985485) ) ) {
              result[0] += 0.025970524973905063;
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.157511234283449042) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.923617362976075107) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += 0.01348589652823275;
                  } else {
                    result[0] += 0.050150626440754645;
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.499747991561890537) ) ) {
                    result[0] += -0.1104275600617233;
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += 0.018793933864627837;
                    } else {
                      result[0] += -0.0015801458255915318;
                    }
                  }
                }
              } else {
                result[0] += -0.02198841716686495;
              }
            }
          } else {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.223051309585572177) ) ) {
              result[0] += -0.09471506745378645;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                result[0] += 0.04038742048288357;
              } else {
                result[0] += -0.01669354794095383;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
          result[0] += -0.008144310838904363;
        } else {
          result[0] += 0.14723076449326192;
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00017517966227556815;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.297559976577759233) ) ) {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.424940347671509677) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.169590950012207919) ) ) {
            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.499747991561890537) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.18088722229004084) ) ) {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                    result[0] += -0.0007192155467263964;
                  } else {
                    result[0] += -0.06975894237859857;
                  }
                } else {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    result[0] += -0.0007182389357299593;
                  } else {
                    result[0] += 0.05547986937449895;
                  }
                }
              } else {
                result[0] += -0.05520769764613789;
              }
            } else {
              result[0] += 0.05673613691137144;
            }
          } else {
            result[0] += 0.031316259937529775;
          }
        } else {
          result[0] += 0.10104900060451485;
        }
      } else {
        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.845019578933716708) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.846404790878296787) ) ) {
            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.803987503051758701) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.636499762535095659) ) ) {
                result[0] += 0.039370605750366126;
              } else {
                result[0] += -0.007219822803347896;
              }
            } else {
              if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.02223539352417081) ) ) {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.88192772865295499) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)5120.000000000000909) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.0036630475886113194;
                    } else {
                      if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)3.276966691017151323) ) ) {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += -0.01600474000990778;
                        } else {
                          result[0] += 0.04251067975872114;
                        }
                      } else {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += 0.02340687916916905;
                        } else {
                          result[0] += -0.005218658113459215;
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
                      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.07836545322574183;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
                          result[0] += -0.05652407270463329;
                        } else {
                          result[0] += 0.04708848849610283;
                        }
                      }
                    } else {
                      result[0] += -0.06244977108090164;
                    }
                  }
                } else {
                  result[0] += -0.0648286238652913;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.158761024475098544) ) ) {
                  result[0] += 0.07779786425319596;
                } else {
                  result[0] += 0.015360322729100491;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.75211906433105646) ) ) {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                  result[0] += -0.005017285630408568;
                } else {
                  result[0] += -0.07063352390848542;
                }
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                  result[0] += -0.005219510705163959;
                } else {
                  result[0] += 0.14284503298295145;
                }
              }
            } else {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)2.500000000000000444) ) ) {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.308072090148926669) ) ) {
                    if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                          result[0] += -0.019634996762226478;
                        } else {
                          result[0] += 0.05176333615694089;
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.213027238845826083) ) ) {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.007876561748581283;
                          } else {
                            result[0] += -0.0783668813268956;
                          }
                        } else {
                          result[0] += 0.01517441505277038;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.053883075714113104) ) ) {
                        if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                          result[0] += -0.021437892834610783;
                        } else {
                          if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)8.049745559692384589) ) ) {
                            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                              result[0] += 0.0810006632391223;
                            } else {
                              result[0] += 0.012857363793047078;
                            }
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                              result[0] += -0.05930123022258852;
                            } else {
                              result[0] += 0.01882158774221766;
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.255827426910402167) ) ) {
                          result[0] += 0.07283678128189902;
                        } else {
                          if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                            result[0] += 0.0024428715466033295;
                          } else {
                            result[0] += 0.07033042039528858;
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.611996650695801669) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.030897617340089667) ) ) {
                        if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)2.917405366897583452) ) ) {
                          result[0] += -0.11803127674260523;
                        } else {
                          result[0] += -0.00306793279951863;
                        }
                      } else {
                        result[0] += -0.000770119959296093;
                      }
                    } else {
                      result[0] += -0.0005517192935569688;
                    }
                  }
                } else {
                  result[0] += 0.10527706933835608;
                }
              } else {
                if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)1.868834793567657693) ) ) {
                  result[0] += 0.045765552786620256;
                } else {
                  result[0] += -0.0064508161824008425;
                }
              }
            }
          }
        } else {
          result[0] += 0.03670000787781115;
        }
      }
    } else {
      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
        if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.740319490432739702) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
              result[0] += 0.0040521025302509995;
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.23636198043823331) ) ) {
                result[0] += 0.11290433707923382;
              } else {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                  result[0] += -0.005769283255761603;
                } else {
                  result[0] += 0.06075797116916091;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                result[0] += -0.07532418499315827;
              } else {
                result[0] += -0.004956221369454892;
              }
            } else {
              result[0] += 0.006765167486277753;
            }
          }
        } else {
          result[0] += 0.05681879703774911;
        }
      } else {
        if ( UNLIKELY( !(data[24].missing != -1) || (data[24].fvalue <= (double)137422176256.0000153) ) ) {
          result[0] += -0.09986676578781817;
        } else {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)8.236541748046876776) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.783551692962647373) ) ) {
                  result[0] += -0.011011096401506184;
                } else {
                  result[0] += 0.08267634142973682;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.636744737625122958) ) ) {
                  result[0] += -0.0200936681545601;
                } else {
                  result[0] += -0.09454507089383302;
                }
              }
            } else {
              result[0] += -0.09400724007231465;
            }
          } else {
            result[0] += 0.05077053444460014;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( LIKELY( !(data[31].missing != -1) || (data[31].fvalue <= (double)112.0000000000000142) ) ) {
      result[0] += -0.00045370852861262946;
    } else {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.176905632019043857) ) ) {
        if ( LIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)7.500000000000000888) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.04879737670608763;
            } else {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                result[0] += 0.014887954762903763;
              } else {
                result[0] += -0.006587512948470685;
              }
            }
          } else {
            if ( UNLIKELY( !(data[30].missing != -1) || (data[30].fvalue <= (double)24.00000000000000355) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  result[0] += 0.024291649522836047;
                } else {
                  result[0] += -0.04511379319839003;
                }
              } else {
                result[0] += -0.01241085956517269;
              }
            } else {
              result[0] += -0.0009611248947585605;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.189660549163820136) ) ) {
            if ( UNLIKELY( !(data[29].missing != -1) || (data[29].fvalue <= (double)192.0000000000000284) ) ) {
              result[0] += -0.023572377510046524;
            } else {
              result[0] += 0.0019704769701832307;
            }
          } else {
            if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)80.00000000000001421) ) ) {
                result[0] += 0.0012542039598670246;
              } else {
                result[0] += -0.02424046725935827;
              }
            } else {
              result[0] += -0.04345483443228814;
            }
          }
        }
      } else {
        result[0] += 0.0022177713932965207;
      }
    }
  } else {
    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.0013155937194842) ) ) {
      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.700598716735840066) ) ) {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.012675821781158891) ) ) {
          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
              result[0] += -0.0010174073264425446;
            } else {
              if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.493027687072754794) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.659457921981812412) ) ) {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += 0.0730878524406223;
                  } else {
                    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.966960191726685458) ) ) {
                      result[0] += 0.03509620459800707;
                    } else {
                      result[0] += -0.05995417376093362;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    result[0] += 0.08059534368816479;
                  } else {
                    result[0] += -0.01095792131383381;
                  }
                }
              } else {
                if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                  result[0] += 0.005528061916029442;
                } else {
                  result[0] += -0.06926745060792205;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                  if ( LIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)2150.000000000000455) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
                      result[0] += -0.03326966594884254;
                    } else {
                      result[0] += 0.034754469512426264;
                    }
                  } else {
                    result[0] += 0.056149196084565714;
                  }
                } else {
                  if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                    result[0] += -0.07898299613206027;
                  } else {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.58491539955139249) ) ) {
                      result[0] += -0.001957848132205192;
                    } else {
                      result[0] += -0.09505653233255341;
                    }
                  }
                }
              } else {
                if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.418317794799805576) ) ) {
                  if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.01029459298026209;
                  } else {
                    if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.384246587753296343) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.86655306816101163) ) ) {
                        if ( LIKELY( !(data[27].missing != -1) || (data[27].fvalue <= (double)2.500000000000000444) ) ) {
                          result[0] += 0.009372965529959799;
                        } else {
                          result[0] += 0.05602675393008551;
                        }
                      } else {
                        result[0] += 0.05621765414138046;
                      }
                    } else {
                      result[0] += 0.06924805362552214;
                    }
                  }
                } else {
                  result[0] += 0.054646065454840034;
                }
              }
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.767332553863526279) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.036931514739991123) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.796801328659058505) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += 0.0757357640817119;
                    } else {
                      if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += 0.04299629129601672;
                      } else {
                        result[0] += -0.013820339821310574;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += -0.016197017079351932;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.439939022064210761) ) ) {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += 0.1152641339229249;
                          } else {
                            result[0] += 0.03737299041417837;
                          }
                        } else {
                          result[0] += -0.005635657098542627;
                        }
                      }
                    } else {
                      result[0] += -0.011901915617845702;
                    }
                  }
                } else {
                  result[0] += -0.0051127829340571335;
                }
              } else {
                result[0] += -0.03350317093997973;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)4.15100884437561124) ) ) {
            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
              result[0] += -0.08578601979522135;
            } else {
              if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.242453336715698464) ) ) {
                result[0] += -0.07426595477381485;
              } else {
                result[0] += -0.009645146672522158;
              }
            }
          } else {
            result[0] += 0.04111252964696865;
          }
        }
      } else {
        if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
            if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                result[0] += -0.040861363884696356;
              } else {
                result[0] += 0.05314996649783278;
              }
            } else {
              result[0] += -0.10402147138841669;
            }
          } else {
            if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.524927973747253862) ) ) {
              result[0] += 0.022382506678700245;
            } else {
              result[0] += -0.03830754920192018;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
            if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  result[0] += 0.07215251852446357;
                } else {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                    result[0] += 0.0644006520397925;
                  } else {
                    result[0] += -0.017561244462087535;
                  }
                }
              } else {
                result[0] += -0.008906725350430097;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.744781017303467685) ) ) {
                result[0] += 0.12322792614522518;
              } else {
                result[0] += 0.037240331157736326;
              }
            }
          } else {
            result[0] += -0.017258032296561567;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
          result[0] += 0.0063642002464643875;
        } else {
          result[0] += 0.1279110200807895;
        }
      } else {
        result[0] += -0.02197032898320448;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    if ( UNLIKELY( !(data[35].missing != -1) || (data[35].fvalue <= (double)1.500000000000000222) ) ) {
      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.944137096405030185) ) ) {
        if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)13.86392068862915217) ) ) {
          if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)1.497866153717041238) ) ) {
            if ( LIKELY( !(data[9].missing != -1) || (data[9].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.83629941940307706) ) ) {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.274755001068116123) ) ) {
                  if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                    if ( UNLIKELY( !(data[36].missing != -1) || (data[36].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                        result[0] += 0.029148597486896195;
                      } else {
                        result[0] += -0.00875878030198239;
                      }
                    } else {
                      result[0] += 0.0037894787413542;
                    }
                  } else {
                    result[0] += 0.005307811889252428;
                  }
                } else {
                  if ( UNLIKELY( !(data[39].missing != -1) || (data[39].fvalue <= (double)3.000000000000000444) ) ) {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.443328142166138583) ) ) {
                      result[0] += 0.019854438952785766;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
                        result[0] += 0.007245204119972495;
                      } else {
                        result[0] += -0.04397914921478494;
                      }
                    }
                  } else {
                    result[0] += 0.0004948120370588074;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                  result[0] += 0.0010034479044091214;
                } else {
                  if ( LIKELY( !(data[33].missing != -1) || (data[33].fvalue <= (double)24.00000000000000355) ) ) {
                    if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.007483005523683417) ) ) {
                        result[0] += -0.09530148977869586;
                      } else {
                        result[0] += -0.023086467029798115;
                      }
                    } else {
                      if ( UNLIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.00000001800250948e-35) ) ) {
                        result[0] += -0.07124159810859175;
                      } else {
                        if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.587220668792725498) ) ) {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.459136486053468573) ) ) {
                            result[0] += 0.0027769267534784154;
                          } else {
                            if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.881510615348816362) ) ) {
                              result[0] += 0.03906902535927312;
                            } else {
                              result[0] += -0.01587160593957967;
                            }
                          }
                        } else {
                          if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.587220668792725498) ) ) {
                            result[0] += -0.005646713690684323;
                          } else {
                            if ( UNLIKELY(  (data[32].missing != -1) && (data[32].fvalue <= (double)-1.00000001800250948e-35) ) ) {
                              result[0] += 0.03800808213293761;
                            } else {
                              result[0] += -0.041285906154272425;
                            }
                          }
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)100.0000000000000142) ) ) {
                      result[0] += -0.015481877944007;
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.373224258422853339) ) ) {
                        result[0] += 0.009795028930448946;
                      } else {
                        result[0] += -0.005991246576612029;
                      }
                    }
                  }
                }
              }
            } else {
              result[0] += 0.002878038562225695;
            }
          } else {
            result[0] += 0.03553213976885607;
          }
        } else {
          result[0] += -0.016666002612077607;
        }
      } else {
        result[0] += -0.019037988921138067;
      }
    } else {
      result[0] += 0.00016092866869958959;
    }
  } else {
    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.08706855773925959) ) ) {
      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.673553824424744096) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)2.962127923965454546) ) ) {
          if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.449861526489258257) ) ) {
            if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.790659427642823154) ) ) {
              if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.636499762535095659) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.720208644866944248) ) ) {
                    result[0] += 0.02591529717777928;
                  } else {
                    result[0] += -0.035669366198036324;
                  }
                } else {
                  result[0] += -0.06480582034812564;
                }
              } else {
                if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)4.569529533386231357) ) ) {
                  if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.773543357849121982) ) ) {
                    result[0] += -0.008983188082802653;
                  } else {
                    if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
                      result[0] += -0.03979718971210316;
                    } else {
                      if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                          result[0] += -0.00927472509411667;
                        } else {
                          result[0] += 0.04778889655248958;
                        }
                      } else {
                        if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.242453336715698464) ) ) {
                          result[0] += 0.11036664336448401;
                        } else {
                          result[0] += 0.009573805134445803;
                        }
                      }
                    }
                  }
                } else {
                  if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.86655306816101163) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.002906363628521324;
                    } else {
                      if ( LIKELY( !(data[10].missing != -1) || (data[10].fvalue <= (double)1.497866153717041238) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.610357046127320224) ) ) {
                          if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.384246587753296343) ) ) {
                            result[0] += 0.010676617082073495;
                          } else {
                            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
                              result[0] += -0.082702094552749;
                            } else {
                              result[0] += 0.055798550264523065;
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                            result[0] += 0.057012710786220215;
                          } else {
                            result[0] += -0.013971763402162087;
                          }
                        }
                      } else {
                        result[0] += -0.0634036812423177;
                      }
                    }
                  } else {
                    result[0] += 0.03023194929852882;
                  }
                }
              }
            } else {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.625595092773438388) ) ) {
                result[0] += 0.02681634361419734;
              } else {
                result[0] += 0.12239702319073507;
              }
            }
          } else {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.05308599308013476;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.134879350662232333) ) ) {
                if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  result[0] += 0.004767530672896235;
                } else {
                  result[0] += -0.005249992617458978;
                }
              } else {
                result[0] += -0.02805784482101589;
              }
            }
          }
        } else {
          if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.388237953186036044) ) ) {
            result[0] += 0.023114437130172014;
          } else {
            result[0] += -0.06423309903100483;
          }
        }
      } else {
        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.861792564392090288) ) ) {
          result[0] += -0.041130921952755334;
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.000308036804200107) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.8958797454833985485) ) ) {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                result[0] += 0.16765465497951865;
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.453179836273194248) ) ) {
                  if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)3.349750161170959917) ) ) {
                    result[0] += -0.09643514431740395;
                  } else {
                    if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.803987503051758701) ) ) {
                      result[0] += -0.027961944426101545;
                    } else {
                      result[0] += 0.13718538925574042;
                    }
                  }
                } else {
                  result[0] += 0.018758704778778255;
                }
              }
            } else {
              result[0] += -0.0028399128885512187;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += -0.046835393474847165;
            } else {
              result[0] += 0.11629831100253307;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.737386107444763628) ) ) {
          result[0] += 0.0018708623615433803;
        } else {
          result[0] += 0.12097687157701034;
        }
      } else {
        result[0] += -0.02028398898277739;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.0001551538774979501;
  } else {
    if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.08706855773925959) ) ) {
      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
        if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)3.43450713157653853) ) ) {
          if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
            result[0] += 0.0014487592980940054;
          } else {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.305786132812500888) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                    if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.350240230560303178) ) ) {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.998158693313599077) ) ) {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
                          result[0] += -0.07519316105650731;
                        } else {
                          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            result[0] += -0.03816121258301146;
                          } else {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.529265403747559482) ) ) {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.003838300704956943) ) ) {
                                result[0] += 0.03352587758856195;
                              } else {
                                result[0] += -0.09161429459323744;
                              }
                            } else {
                              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.920663833618164951) ) ) {
                                result[0] += 0.11123928274966313;
                              } else {
                                result[0] += 0.014481411376067286;
                              }
                            }
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.03420138359069913) ) ) {
                          result[0] += 0.107981662147365;
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.803987503051758701) ) ) {
                            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.744487762451173651) ) ) {
                              result[0] += -0.09826935600715075;
                            } else {
                              result[0] += 0.015011441846410982;
                            }
                          } else {
                            result[0] += 0.04653320373973711;
                          }
                        }
                      }
                    } else {
                      result[0] += 0.005949642277118238;
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.098348140716553623) ) ) {
                      result[0] += 0.10462154652954647;
                    } else {
                      result[0] += -0.001301831456662525;
                    }
                  }
                } else {
                  result[0] += -0.10909836458393614;
                }
              } else {
                if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
                  if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += 0.07805503273926107;
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.68799614906311124) ) ) {
                      result[0] += 0.04386904960729638;
                    } else {
                      result[0] += -0.07465689448412127;
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
                    result[0] += -0.05493869432473664;
                  } else {
                    if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)5.503294944763184482) ) ) {
                        result[0] += 0.0113974921121454;
                      } else {
                        result[0] += -0.1398275706081452;
                      }
                    } else {
                      result[0] += -0.19182692291174366;
                    }
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.917705297470093662) ) ) {
                  result[0] += 0.028210965136531943;
                } else {
                  result[0] += -0.05773608761780661;
                }
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.494428873062134677) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.003838300704956943) ) ) {
                            result[0] += 0.10404592670676482;
                          } else {
                            result[0] += 0.044916105082646395;
                          }
                        } else {
                          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.268911361694336826) ) ) {
                            if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.729812622070313388) ) ) {
                              result[0] += -0.06152112425972367;
                            } else {
                              result[0] += 0.024995537657786424;
                            }
                          } else {
                            result[0] += 0.03692545537380833;
                          }
                        }
                      } else {
                        result[0] += -0.021933379260229854;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.942744255065918857) ) ) {
                        result[0] += -0.10767661623533016;
                      } else {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.658699750900269443) ) ) {
                            result[0] += -0.051875338234379544;
                          } else {
                            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += -0.016319092067620843;
                            } else {
                              if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.385823249816895419) ) ) {
                                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.138333082199097124) ) ) {
                                  if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                                    result[0] += -0.053709430587803156;
                                  } else {
                                    result[0] += 0.006439150829854618;
                                  }
                                } else {
                                  if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.670753479003907138) ) ) {
                                      result[0] += 0.08470180425573184;
                                    } else {
                                      result[0] += 0.020535270015035873;
                                    }
                                  } else {
                                    result[0] += -0.024369386539785202;
                                  }
                                }
                              } else {
                                result[0] += 0.06182642179343705;
                              }
                            }
                          }
                        } else {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.339535951614380771) ) ) {
                            result[0] += 0.046693153623889774;
                          } else {
                            result[0] += -0.0051184048068478065;
                          }
                        }
                      }
                    }
                  } else {
                    result[0] += -0.012558550762005639;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.768316030502320224) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.729812622070313388) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
                          if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
                            result[0] += -0.14841761723009894;
                          } else {
                            result[0] += -0.0013472925456123066;
                          }
                        } else {
                          result[0] += -0.2170554069228668;
                        }
                      } else {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.601370334625245029) ) ) {
                          result[0] += 0.08346267154897834;
                        } else {
                          result[0] += -0.04923163785780647;
                        }
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.797939777374268466) ) ) {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.242453336715698464) ) ) {
                            result[0] += -0.14077847976129648;
                          } else {
                            result[0] += 0.022060067315751872;
                          }
                        } else {
                          result[0] += 0.09143366461694843;
                        }
                      } else {
                        if ( UNLIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          result[0] += -0.0014206352969609421;
                        } else {
                          result[0] += -0.09426638607450698;
                        }
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.930492877960205966) ) ) {
                      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += 0.04566401595365496;
                      } else {
                        result[0] += 0.0005666717204405456;
                      }
                    } else {
                      result[0] += -0.03111686303619047;
                    }
                  }
                }
              }
            }
          }
        } else {
          result[0] += 0.04195343846388955;
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.257356405258179155) ) ) {
          if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.00000001800250948e-35) ) ) {
            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
              result[0] += 0.02789598909332719;
            } else {
              result[0] += -0.019474753141800097;
            }
          } else {
            result[0] += 0.07485170406282471;
          }
        } else {
          result[0] += -0.013792401029625019;
        }
      }
    } else {
      if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
        result[0] += 0.005285564846060798;
      } else {
        result[0] += 0.12624083341422987;
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.0001614110055339938;
  } else {
    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
      if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)2.861792564392090288) ) ) {
        if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)11.0013155937194842) ) ) {
          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.726826429367066318) ) ) {
            result[0] += -0.00023655371003175737;
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.802100181579590732) ) ) {
              if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += 0.04991522433845141;
              } else {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.847910165786744052) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.611996650695801669) ) ) {
                    result[0] += 0.0033747100047232005;
                  } else {
                    result[0] += -0.09615437819998912;
                  }
                } else {
                  result[0] += 0.06644616816571446;
                }
              }
            } else {
              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += 0.057627231685453285;
              } else {
                if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                  if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.03383037189348245;
                  } else {
                    result[0] += 0.01690425736678418;
                  }
                } else {
                  result[0] += -0.07982993723283313;
                }
              }
            }
          }
        } else {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.569529533386231357) ) ) {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)11.85305833816528498) ) ) {
              result[0] += 0.08286125691345611;
            } else {
              result[0] += 0.0009189854596718426;
            }
          } else {
            result[0] += 0.12526358174916383;
          }
        }
      } else {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
          if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.257356405258179155) ) ) {
            result[0] += -0.02448641007350806;
          } else {
            result[0] += -0.10391764433944216;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
              if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                result[0] += -0.0257936172133008;
              } else {
                result[0] += 0.041579640120061004;
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.493027687072754794) ) ) {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.923617362976075107) ) ) {
                  result[0] += -0.053514940537721406;
                } else {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.847910165786744052) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                        result[0] += -0.0017078778576242065;
                      } else {
                        result[0] += -0.0786900993732549;
                      }
                    } else {
                      if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                        result[0] += -0.019090964402307856;
                      } else {
                        result[0] += 0.08236438559056924;
                      }
                    }
                  } else {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)0.8958797454833985485) ) ) {
                      result[0] += 0.07737897949045179;
                    } else {
                      if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                        result[0] += -0.0662103483251742;
                      } else {
                        result[0] += 0.007268176375416213;
                      }
                    }
                  }
                }
              } else {
                result[0] += 0.013348083596724632;
              }
            }
          } else {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.674522399902344638) ) ) {
              if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.51675081253051935) ) ) {
                result[0] += 0.001463457535514386;
              } else {
                result[0] += -0.01399387120213096;
              }
            } else {
              if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                result[0] += -0.02525057912187652;
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.272946834564209873) ) ) {
                  if ( LIKELY( !(data[21].missing != -1) || (data[21].fvalue <= (double)112.0000000000000142) ) ) {
                    result[0] += 0.01580024106372078;
                  } else {
                    result[0] += 0.11349335984851877;
                  }
                } else {
                  result[0] += -0.011185620412176715;
                }
              }
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.511434078216553178) ) ) {
          result[0] += 0.04662200612618107;
        } else {
          if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.418317794799805576) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.566809177398682529) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.158761024475098544) ) ) {
                  if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.126503944396974433) ) ) {
                    result[0] += -0.11010649082899633;
                  } else {
                    result[0] += -0.0038166594588890687;
                  }
                } else {
                  result[0] += -0.011522757272050285;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.744568347930909091) ) ) {
                  result[0] += 0.1006004963980449;
                } else {
                  result[0] += -0.041308295107903785;
                }
              }
            } else {
              result[0] += 0.07079848556620368;
            }
          } else {
            if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)2.012675821781158891) ) ) {
              result[0] += 0.013729318623346535;
            } else {
              result[0] += -0.20993039669295116;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.659457921981812412) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)1.700598716735840066) ) ) {
            result[0] += -0.04201306168212734;
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)10.42895507812500178) ) ) {
              result[0] += 0.026192905415576964;
            } else {
              result[0] += -0.006705697030445346;
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
            if ( UNLIKELY( !(data[23].missing != -1) || (data[23].fvalue <= (double)1900.000000000000227) ) ) {
              result[0] += -0.05031831238647371;
            } else {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                if ( LIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
                  if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.333273410797120029) ) ) {
                      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.673553824424744096) ) ) {
                        result[0] += 0.0036108184226851265;
                      } else {
                        result[0] += 0.09805144281943043;
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.90474271774292081) ) ) {
                        result[0] += 0.002731904617726194;
                      } else {
                        if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                          result[0] += 0.06980443235284042;
                        } else {
                          if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
                            if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                              result[0] += 0.012827622721286744;
                            } else {
                              result[0] += 0.10534903250793648;
                            }
                          } else {
                            result[0] += -0.004515320414916664;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                      result[0] += 0.0021769692897662374;
                    } else {
                      result[0] += 0.03323721595592654;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.134879350662232333) ) ) {
                    result[0] += -0.025369023834685867;
                  } else {
                    result[0] += 0.02644305706919016;
                  }
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.884705543518067294) ) ) {
                  if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.497866153717041238) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += 0.026033343913877724;
                    } else {
                      result[0] += -0.03677465498456214;
                    }
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.004448907493353353;
                    } else {
                      result[0] += 0.0735410360208455;
                    }
                  }
                } else {
                  if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.276966691017151323) ) ) {
                    result[0] += -0.04586307186412683;
                  } else {
                    result[0] += 0.02223043753817593;
                  }
                }
              }
            }
          } else {
            result[0] += -0.011482442296939092;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00013621486212811687;
  } else {
    if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
      if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
        if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
          result[0] += 0.05788702595904696;
        } else {
          if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
            result[0] += -0.08623040262345921;
          } else {
            if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
              if ( UNLIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)5.668153762817383701) ) ) {
                result[0] += 0.0739012983859498;
              } else {
                result[0] += -0.04734758783500909;
              }
            } else {
              result[0] += -0.06862478649891966;
            }
          }
        }
      } else {
        if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.00000001800250948e-35) ) ) {
          if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.868834793567657693) ) ) {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.628996372222901279) ) ) {
              result[0] += -0.007840107018024617;
            } else {
              result[0] += -0.09336225282338148;
            }
          } else {
            result[0] += 0.04692209129721267;
          }
        } else {
          if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)3.500000000000000444) ) ) {
            if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.602003335952759233) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.257356405258179155) ) ) {
                result[0] += 0.12096011350928998;
              } else {
                if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.917405366897583452) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.438968896865845615) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.0058673036570496846;
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.933422565460205966) ) ) {
                          result[0] += 0.04266917241004515;
                        } else {
                          result[0] += -0.04087214888021271;
                        }
                      } else {
                        result[0] += 0.054953007039529746;
                      }
                    }
                  } else {
                    result[0] += 0.046151443401968945;
                  }
                } else {
                  result[0] += -0.004346109294486639;
                }
              }
            } else {
              result[0] += 0.002485668268187788;
            }
          } else {
            if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.259080410003662998) ) ) {
              if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.315590381622315341) ) ) {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.087577104568482333) ) ) {
                      if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)4.579839229583741123) ) ) {
                        result[0] += 0.04374298832722552;
                      } else {
                        result[0] += -0.04522751176627401;
                      }
                    } else {
                      if ( LIKELY( !(data[6].missing != -1) || (data[6].fvalue <= (double)1.242453336715698464) ) ) {
                        result[0] += 0.008942797716970324;
                      } else {
                        result[0] += -0.10107250572085102;
                      }
                    }
                  } else {
                    result[0] += -0.05727559621403192;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.511434078216553178) ) ) {
                      result[0] += -0.0641112666459574;
                    } else {
                      if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.921924352645874468) ) ) {
                        result[0] += 0.08032795556394509;
                      } else {
                        result[0] += -0.04052412217463689;
                      }
                    }
                  } else {
                    result[0] += -0.028672703689376012;
                  }
                }
              } else {
                result[0] += -0.012215242439968396;
              }
            } else {
              result[0] += -0.03631694064351974;
            }
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.384246587753296343) ) ) {
        if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.138333082199097124) ) ) {
          result[0] += -0.020444385784656082;
        } else {
          result[0] += 0.01026271707447991;
        }
      } else {
        if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)5.000579357147217685) ) ) {
          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.610357046127320224) ) ) {
            if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)2.500000000000000444) ) ) {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.453179836273194248) ) ) {
                if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)1.700598716735840066) ) ) {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.303973913192749912) ) ) {
                    if ( UNLIKELY( !(data[45].missing != -1) || (data[45].fvalue <= (double)1.500000000000000222) ) ) {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.119004011154175693) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.23636198043823331) ) ) {
                          result[0] += 0.011296608337619847;
                        } else {
                          result[0] += -0.06474626496085858;
                        }
                      } else {
                        result[0] += -0.08497901882533057;
                      }
                    } else {
                      result[0] += -0.001890744468158797;
                    }
                  } else {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.438968896865845615) ) ) {
                        if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.797939777374268466) ) ) {
                          if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.868834793567657693) ) ) {
                            result[0] += -0.01186793512220051;
                          } else {
                            result[0] += 0.045889974220289426;
                          }
                        } else {
                          result[0] += 0.06949458018628456;
                        }
                      } else {
                        result[0] += -0.04531453196069856;
                      }
                    } else {
                      if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.400584220886231357) ) ) {
                        result[0] += 0.06975040585294869;
                      } else {
                        if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)4.863673448562622958) ) ) {
                          result[0] += -0.04566703957129652;
                        } else {
                          if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)1.497866153717041238) ) ) {
                            if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.075335502624512607) ) ) {
                              result[0] += 0.07371032634266879;
                            } else {
                              result[0] += -0.009949160537810071;
                            }
                          } else {
                            result[0] += -0.018375803167588217;
                          }
                        }
                      }
                    }
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)2.770631790161133257) ) ) {
                    if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.481121778488159624) ) ) {
                      if ( LIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.802901029586792436) ) ) {
                        result[0] += 0.03695155157114234;
                      } else {
                        result[0] += -0.04051245634443476;
                      }
                    } else {
                      result[0] += -0.025318731731974117;
                    }
                  } else {
                    if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.177185058593750444) ) ) {
                      if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                        result[0] += -0.00501828251048489;
                      } else {
                        result[0] += -0.06632082228897652;
                      }
                    } else {
                      result[0] += 0.006440679262108693;
                    }
                  }
                }
              } else {
                result[0] += 0.00038312157609916636;
              }
            } else {
              if ( UNLIKELY( !(data[13].missing != -1) || (data[13].fvalue <= (double)2.970085620880127397) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.339535951614380771) ) ) {
                  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.802901029586792436) ) ) {
                    if ( LIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)5.645421981811524326) ) ) {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.729812622070313388) ) ) {
                        result[0] += -0.020451296387427556;
                      } else {
                        result[0] += 0.030924053190221804;
                      }
                    } else {
                      result[0] += 0.06631744421543397;
                    }
                  } else {
                    result[0] += -0.031037041762154383;
                  }
                } else {
                  result[0] += -0.055258482309053995;
                }
              } else {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.438968896865845615) ) ) {
                  result[0] += -0.05121301029095981;
                } else {
                  result[0] += 0.10314154873702629;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.837713479995728427) ) ) {
              result[0] += 0.03395964630172399;
            } else {
              if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)2.970085620880127397) ) ) {
                if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)8.236541748046876776) ) ) {
                  result[0] += 0.004912469343555492;
                } else {
                  result[0] += 0.052262227850407066;
                }
              } else {
                result[0] += -0.05919914139505274;
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[15].missing != -1) || (data[15].fvalue <= (double)3.067782521247864214) ) ) {
            result[0] += 0.1045910741351698;
          } else {
            result[0] += 0.0036068311300925596;
          }
        }
      }
    }
  }
  if ( LIKELY( !(data[32].missing != -1) || (data[32].fvalue <= (double)96.00000000000001421) ) ) {
    result[0] += -0.00013774519365407128;
  } else {
    if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)7.86655306816101163) ) ) {
      if ( LIKELY( !(data[44].missing != -1) || (data[44].fvalue <= (double)1.00000001800250948e-35) ) ) {
        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.420525312423706943) ) ) {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)8.318498134613038886) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.799612998962403232) ) ) {
              if ( LIKELY( !(data[16].missing != -1) || (data[16].fvalue <= (double)1.242453336715698464) ) ) {
                if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)5.138696432113648349) ) ) {
                  if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                    result[0] += -0.062091277751660234;
                  } else {
                    result[0] += -0.008180011495678693;
                  }
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.297559976577759233) ) ) {
                    result[0] += 0.04058645200396263;
                  } else {
                    result[0] += -0.04214429289078412;
                  }
                }
              } else {
                if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.382196187973023349) ) ) {
                  if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)5.036931514739991123) ) ) {
                    if ( LIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      if ( LIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)1.497866153717041238) ) ) {
                        result[0] += -0.022601918573075242;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)3.737386107444763628) ) ) {
                          result[0] += -0.02120216184908164;
                        } else {
                          result[0] += 0.053119436444001446;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)3.465247392654419389) ) ) {
                        result[0] += -0.04268436209702506;
                      } else {
                        if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.060294389724732333) ) ) {
                          result[0] += 0.06016741756963976;
                        } else {
                          result[0] += 0.008529375545235062;
                        }
                      }
                    }
                  } else {
                    result[0] += 0.054231521992978884;
                  }
                } else {
                  if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)4.731793165206910068) ) ) {
                    result[0] += 0.02976668409300151;
                  } else {
                    result[0] += -0.026770760057314547;
                  }
                }
              }
            } else {
              if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)4.189540147781372958) ) ) {
                result[0] += 0.08414146586081414;
              } else {
                if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.932935476303101474) ) ) {
                  result[0] += 0.06303186767435144;
                } else {
                  result[0] += -0.0010861870680298781;
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
              if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.23832273483276456) ) ) {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.129780292510988104) ) ) {
                  if ( LIKELY( !(data[8].missing != -1) || (data[8].fvalue <= (double)0.8958797454833985485) ) ) {
                    if ( UNLIKELY( !(data[43].missing != -1) || (data[43].fvalue <= (double)1.00000001800250948e-35) ) ) {
                      result[0] += -0.05660660013772187;
                    } else {
                      result[0] += 0.05058366154560737;
                    }
                  } else {
                    result[0] += 0.08508917490333326;
                  }
                } else {
                  result[0] += -0.03167314381119127;
                }
              } else {
                result[0] += 0.04699873401617147;
              }
            } else {
              if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)2.012675821781158891) ) ) {
                result[0] += -0.12392726849883728;
              } else {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.249904870986938921) ) ) {
                  if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)3.569433569908142534) ) ) {
                    result[0] += 0.03024862427626924;
                  } else {
                    result[0] += 0.007820043049092354;
                  }
                } else {
                  result[0] += 0.09880279246285126;
                }
              }
            }
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.427738666534424716) ) ) {
            if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.772996187210083896) ) ) {
              if ( LIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)1.700598716735840066) ) ) {
                if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)1.497866153717041238) ) ) {
                  if ( LIKELY( !(data[22].missing != -1) || (data[22].fvalue <= (double)7168.000000000000909) ) ) {
                    if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.678428173065186435) ) ) {
                      if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.543205261230469638) ) ) {
                        result[0] += 0.0075108141381874495;
                      } else {
                        result[0] += -0.03732413751335089;
                      }
                    } else {
                      if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.689592361450196201) ) ) {
                        if ( LIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)280.0000000000000568) ) ) {
                          if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.248013019561768466) ) ) {
                            if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.357556104660035068) ) ) {
                              result[0] += 0.0028349826161480148;
                            } else {
                              if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)3.156774044036865678) ) ) {
                                result[0] += 0.08514665346426298;
                              } else {
                                result[0] += -0.0040659833201221116;
                              }
                            }
                          } else {
                            result[0] += -0.03161919721254796;
                          }
                        } else {
                          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)3.624251961708069292) ) ) {
                            result[0] += 0.0809218838457787;
                          } else {
                            result[0] += 0.017443160272490234;
                          }
                        }
                      } else {
                        if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.080862283706665927) ) ) {
                          if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)6.966960191726685458) ) ) {
                            result[0] += 0.02737249532950183;
                          } else {
                            result[0] += 0.06314589226253535;
                          }
                        } else {
                          if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)7.126885652542115146) ) ) {
                            result[0] += -0.04316655828252474;
                          } else {
                            result[0] += 0.018404360967675185;
                          }
                        }
                      }
                    }
                  } else {
                    if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)6.650908708572388583) ) ) {
                      result[0] += 0.08379425968474587;
                    } else {
                      if ( LIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)5.223051309585572177) ) ) {
                        result[0] += 0.023940625794967637;
                      } else {
                        result[0] += -0.09511979881656202;
                      }
                    }
                  }
                } else {
                  result[0] += 0.07951496347050381;
                }
              } else {
                result[0] += -0.07542933802077906;
              }
            } else {
              if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)2.249904870986938921) ) ) {
                if ( UNLIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)6.333273410797120029) ) ) {
                  result[0] += 0.08291692848054172;
                } else {
                  result[0] += -0.006120707169263072;
                }
              } else {
                if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.597218394279480425) ) ) {
                  result[0] += -0.050382721910322094;
                } else {
                  if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.108135223388672763) ) ) {
                    if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.500000000000000222) ) ) {
                      result[0] += -0.021878409908555094;
                    } else {
                      result[0] += 0.048948843014195983;
                    }
                  } else {
                    if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)6.358708143234253818) ) ) {
                      if ( LIKELY( !(data[17].missing != -1) || (data[17].fvalue <= (double)6.296187639236451083) ) ) {
                        result[0] += 0.034979600295618384;
                      } else {
                        if ( LIKELY( !(data[26].missing != -1) || (data[26].fvalue <= (double)1.00000001800250948e-35) ) ) {
                          if ( LIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)7.30853915214538663) ) ) {
                            result[0] += -0.045882899021288656;
                          } else {
                            result[0] += 0.0369433809434792;
                          }
                        } else {
                          result[0] += 0.04857710288418967;
                        }
                      }
                    } else {
                      if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)5.972535848617554599) ) ) {
                        if ( LIKELY( !(data[14].missing != -1) || (data[14].fvalue <= (double)7.890938758850098544) ) ) {
                          result[0] += 0.0032554030745656003;
                        } else {
                          result[0] += 0.0670195244569539;
                        }
                      } else {
                        result[0] += -0.020223721980481585;
                      }
                    }
                  }
                }
              }
            }
          } else {
            if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)9.704609394073488104) ) ) {
              result[0] += -0.005374041529554134;
            } else {
              if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)2.861792564392090288) ) ) {
                result[0] += 0.006704554056534619;
              } else {
                result[0] += 0.04900097069085026;
              }
            }
          }
        }
      } else {
        if ( LIKELY( !(data[12].missing != -1) || (data[12].fvalue <= (double)0.8958797454833985485) ) ) {
          if ( UNLIKELY( !(data[19].missing != -1) || (data[19].fvalue <= (double)180.0000000000000284) ) ) {
            result[0] += -0.028131278856308818;
          } else {
            result[0] += 0.0009134532752951787;
          }
        } else {
          if ( UNLIKELY( !(data[18].missing != -1) || (data[18].fvalue <= (double)3.921924352645874468) ) ) {
            result[0] += -0.12944530597169207;
          } else {
            result[0] += -0.02146064685971325;
          }
        }
      }
    } else {
      result[0] += -0.02670379861527999;
    }
  }
}

